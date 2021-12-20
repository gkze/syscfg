#!/usr/bin/env python3
"""syscfg is a lightweight system configurator with a declarative API."""
from __future__ import annotations

import logging
import multiprocessing
import os
import platform
import re
import selectors
import shutil
import subprocess
import sys
from abc import ABC
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from enum import Enum, auto
from functools import cached_property, partial
from io import StringIO
from logging import Logger
from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Queue
from selectors import DefaultSelector
from threading import Thread
from typing import (
    IO,
    Any,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    NewType,
    NoReturn,
    Sequence,
    Union,
    cast,
)

LogLevel = NewType("LogLevel", int)
MacOSDefaultValue = Union[str, bool, float, int, "list[Any]"]
PathLike = Union[Path, str]
BytesOrTextIO = Union[IO[bytes], IO[str]]

CPU_COUNT: int = multiprocessing.cpu_count()
PROGNAME: str = "syscfg"
HOMEBREW_INSTALL_URL: str = (
    "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
)

LOG_FMT: str = "%(asctime)s [%(levelname)s] [%(name)s] %(msg)s"
LOGGER: Logger = logging.getLogger(f"{PROGNAME}.main")


def log_cmd_streams(
    cmd: Sequence[Any],
    env: Mapping[str, str] | None = os.environ,
    stdin: BytesOrTextIO | None = None,
    logger: Logger = LOGGER,
    stdout_lvl: LogLevel = cast(LogLevel, logging.INFO),
    stderr_lvl: LogLevel = cast(LogLevel, logging.INFO),
) -> int:
    """
    Utility method to log data from the stdout and stderr streams of a command.
    Returns the exit code of the command when the process terminates.
    """
    proc: subprocess.Popen = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    sel: DefaultSelector = DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)  # type: ignore
    sel.register(proc.stderr, selectors.EVENT_READ)  # type: ignore

    while True:
        for key, _ in sel.select():
            data = key.fileobj.read1().decode()  # type: ignore

            if not data:
                proc.communicate(cast(bytes, stdin))
                return proc.wait()

            lines: Iterator[str] = filter(None, data.split("\n"))

            if key.fileobj is proc.stdout:
                for line in lines:
                    logger.log(stdout_lvl, f"({' '.join(cmd)}) (stdout) {line}")

            else:
                for line in lines:
                    logger.log(stderr_lvl, f"({' '.join(cmd)}) (stderr) {line}")


class Node:
    """Node represents a particular task to execute."""

    # pylint: disable=dangerous-default-value
    def __init__(self: Node, dependencies: set[Node] = set()) -> None:
        """Initialize a task node."""
        self.dependencies: set[Node] = dependencies
        self.dependents: set[Node] = set()

    def __eq__(self: Node, other: object) -> bool:
        return hash(self) == hash(other)

    def __hash__(self: Node) -> int:
        return hash(self._key)

    def __repr__(self: Node) -> str:
        """Representation of the node"""
        return f"<{self.__class__.__name__}>"

    def clear(self: Node) -> None:
        """
        Clear this node from its dependents' set of dependencies, to allow
        for it to be queued.
        """
        for dependent in self.dependents:
            dependent.dependencies = dependent.dependencies - {self}
            self.dependents = self.dependents - {dependent}

    def resolve(self: Node) -> None:
        """
        Resolve this node. This means executing the task the node
        encapsulates and clearing it from its dependents' dependencies.

        This method is called by the task graph processor and is not to be
        overridden.
        """
        self.act()
        self.clear()

    @property
    def _key(self: Node) -> Hashable:
        """
        Key to uniquely identify the object over its lifecycle. Must be
        implemented by subclass to be a valid Node.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._key() must be implemented by subclass!"
        )

    def act(self: Node) -> None:
        """
        The node task. This method needs to be subclassed to describe the
        task actions to perform.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.act() needs to be implemented by subclass!",
        )


def traverse(n: Node, workq: Queue, queued: set[Node], done: set[Node]) -> None:
    """
    Traverse node and queue it if it's dependency-free, otherwise traverse
    its dependency nodes and do the same.
    """
    if n.dependencies == set():
        if n in done:
            n.clear()
            return

        if n not in queued:
            LOGGER.debug(f"Queueing {n}")
            queued.add(n)
            workq.put(n)
            return

    for dep in n.dependencies:
        dep.dependents = dep.dependents | {n}
        traverse(dep, workq, queued, done)


def dispatch(n: Node, workq: Queue, queued: set[Node], done: set[Node]) -> None:
    """
    The dispatch Thread logic. Continuously scan the passed node, queueing
    dependency-free nodes for execution, and returning when the passed node
    (the root node) itself becomes dependency-free (after it gets queued).
    """
    while True:
        traverse(n, workq, queued, done)

        if n.dependencies == set():
            traverse(n, workq, queued, done)
            return


def work(q: Queue, done: set[Node]) -> None:
    """
    Simple main method for workers. Just get nodes from the queue and resolve
    them.
    """
    while True:
        n: Node = cast(Node, q.get(True))
        n.resolve()
        done.add(n)
        q.task_done()


def process(*nodes: Node) -> None:
    """
    Main method for executing task graphs. This does all the heavy lifting of
    managing the lifecycle of the dispatch and worker thread pools.
    """
    workq: Queue = Queue()
    queued: set[Node] = set()
    done: set[Node] = set()

    dispatch_threads: set[Thread] = set()
    for node in nodes:
        # fmt: off
        dispatch_thread: Thread = Thread(
            target=dispatch, args=(node, workq, queued, done), daemon=True,
        )
        # fmt: on
        dispatch_thread.start()
        dispatch_threads.add(dispatch_thread)

    workers: ThreadPool = ThreadPool(CPU_COUNT, work, (workq, done))

    for thread in dispatch_threads:
        thread.join()

    workers.close()
    workq.join()


class Container(Node):
    """
    Container is a special type of Node that has no action. Its only purpose
    is to hold dependencies.
    """

    def __init__(
        self: Container, elements: set[Node] = set(), dependencies: set[Node] = set()
    ) -> None:
        super().__init__(dependencies)
        self._elements: set[Node] = elements

    def __repr__(self: Container) -> str:
        return f"<{self.__class__.__name__} ({len(self._elements)})>"

    @property
    def _key(self: Container) -> Hashable:
        return tuple(self._elements)

    def act(self: Container) -> None:
        process(*self._elements)


class Serial(Container):
    """
    Serial is a special Node that converts the passed sequence of
    nodes into a linked list, which is a type of a unidirectional DAG. This
    allows for explicit node ordering.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self: Serial,
        order: list[Node],
        dependencies: set[Node] = set(),
    ) -> None:
        super().__init__(dependencies)
        self._order: Sequence[Node] = deepcopy(order)

        def add_dependency(n: Node, deps: list[Node]) -> None:
            if not deps:
                return

            current_node: Node = deepcopy(deps.pop())

            n.dependencies = n.dependencies | {current_node}

            add_dependency(current_node, deps)

        add_dependency(self, order)

    def __repr__(self: Serial) -> str:
        return f"<{self.__class__.__name__} ({len(self._order)})>"

    @property
    def _key(self: Serial) -> Hashable:
        return tuple(self._order)

    def act(self: Serial) -> None:
        for dep in self.dependencies:
            process(dep)


class Command(Node):
    """A command to be executed on the system."""

    # pylint: disable=dangerous-default-value
    def __init__(
        self: Command,
        args: Sequence[str | Path],
        stdin: BytesOrTextIO | None = None,
        env: Mapping[str, str] | None = os.environ,
        logger: Logger = LOGGER,
        dependencies: set[Node] = set(),
    ):
        super().__init__(dependencies)

        self._args: Sequence[str] = list(map(str, args))
        self._stdin: BytesOrTextIO | None = stdin
        self._env: Mapping[str, str] | None = env
        self._logger: Logger = logger

    def __repr__(self: Command) -> str:
        return f"<{self.__class__.__name__} {self._args}>"

    @property
    def _key(self: Command) -> Hashable:
        return tuple(self._args)

    def act(self: Command) -> None:
        log_cmd_streams(self._args, env=self._env, logger=self._logger)


class Hostname(Node, ABC):
    def __init__(
        self: Hostname, hostname: str, dependencies: set[Node] = set()
    ) -> None:
        super().__init__(dependencies)

        self._hostname: str = hostname

    def __repr__(self: Hostname) -> str:
        return f"<{self.__class__.__name__} {self._hostname}>"

    @property
    def _key(self: Hostname) -> Hashable:
        return self._hostname

    def _raise_unsupported(self: Hostname, _os: str) -> NoReturn:
        raise RuntimeError(f"{_os} is not supported!")

    def _set_linux_hostname(self: Hostname) -> None:
        if Path("/sbin/init").resolve().name == "systemd":
            process(Command(["hostnamectl", "set-hostname", self._hostname]))
            return

        # fmt: off
        process(Serial([
            File("/etc/hostname", 0o644, contents=f"{self._hostname}\n"),
            File(
                "/etc/hosts",
                0o644,
                FileAction.APPEND,
                contents=f"127.0.0.1 {self._hostname}",
            ),
            Command(["/etc/init.d/hostname", "restart"]),
        ]))
        # fmt: on

    def _set_macos_hostname(self: Hostname) -> None:
        # fmt: off
        process(Serial([
            Container({
                *[
                    Command(["scutil", "--set", hnt, self._hostname])
                    for hnt in {"ComputerName", "HostName", "LocalHostName"}
                ],
                MacOSDefault( "com.apple.smb.server", "NetBIOSName", self._hostname),
            }),
            Container({
                    Command(["dscacheutil", "-flushcache"]),
                    Command(["killall", "-9", "mDNSResponder"]),
            }),
        ]))
        # fmt: on

    def act(self: Hostname) -> None:
        system: str = platform.system()

        {
            "Linux": self._set_linux_hostname,
            "Darwin": self._set_macos_hostname,
        }.get(system, partial(self._raise_unsupported, system))()


class MacOSDefaultValueType(Enum):
    """Represents all possible macOS defaults system value types."""

    ARRAY: str = "array"
    BOOL: str = "bool"
    FLOAT: str = "float"
    INT: str = "int"
    STRING: str = "string"


class MacOSDefault(Node):
    """Sets a macOS default. Currently only supports writing."""

    def __init__(
        self: MacOSDefault,
        domain: str,
        name: str,
        value: MacOSDefaultValue,
        dependencies: set[Node] = set(),
    ) -> None:
        super().__init__(dependencies)

        self._domain: str = domain
        self._name: str = name
        self._value: MacOSDefaultValue = value

    def __repr__(self: MacOSDefault) -> str:
        return (
            f"<{self.__class__.__name__} "
            + f"domain={self._domain} name={self._name} value={self._value}>"
        )

    @property
    def _key(self: MacOSDefault) -> Hashable:
        # fmt: off
        return tuple((
            self._domain,
            self._name,
            self._value if not isinstance(self._value, list)
            else tuple(self._value),
        ))
        # fmt: on

    def act(self: MacOSDefault) -> None:
        # fmt: off
        process(Command(["defaults", "write", self._domain, self._name, *{
            "list": lambda: [
                f"-{MacOSDefaultValueType.ARRAY.value}",
                " ".join(cast(Iterable[str], self._value)),
            ],
            "bool": lambda: [
                f"-{MacOSDefaultValueType.BOOL.value}",
                str(self._value).lower(),
            ],
            "float": lambda: [
                f"-{MacOSDefaultValueType.FLOAT.value}",
                str(self._value),
            ],
            "int": lambda: [
                f"-{MacOSDefaultValueType.INT.value}",
                str(self._value),
            ],
            "str": lambda: [
                f"-{MacOSDefaultValueType.STRING.value}",
                cast(str, self._value),
            ],
        }[type(self._value).__name__]()]))
        # fmt: on


class MacOSFileFlagAction(Enum):
    SET: str = ""
    CLEAR: str = "no"


class MacOSFileFlagType(Enum):
    ARCHIVED: str = "archived"
    OPAQUE: str = "opaque"
    DUMP: str = "dump"
    SAPPEND: str = "sappend"
    SIMMUTABLE: str = "simmutable"
    UAPPEND: str = "uappend"
    UIMMUTABLE: str = "uimmutable"
    HIDDEN: str = "hidden"


class MacOSFileFlag(Node):
    """(Un)Sets macOS-specific file flags."""

    def __init__(
        self: MacOSFileFlag,
        action: MacOSFileFlagAction,
        ftype: MacOSFileFlagType,
        target: Path,
        dependencies: set[Node] = set(),
    ) -> None:
        super().__init__(dependencies)

        self._action: MacOSFileFlagAction = action
        self._ftype: MacOSFileFlagType = ftype
        self._target: Path = target

    @property
    def _key(self: MacOSFileFlag) -> Hashable:
        return (self._action, self._ftype, self._target)

    def __repr__(self: MacOSFileFlag) -> str:
        return (
            f"<{self.__class__.__name__} "
            + f"action={self._action} flagtype={self._ftype} target={self._target}>"
        )

    def act(self: MacOSFileFlag) -> None:
        # fmt: off
        process(Command([
            "chflags",
            f"{self._action.value}{self._ftype.value}", str(self._target),
        ]))
        # fmt: on


class FileAction(Enum):
    """Possible file actions."""

    CREATE: int = auto()
    APPEND: int = auto()
    REMOVE: int = auto()


class FileType(Enum):
    """Possible file types."""

    FILE: int = auto()
    DIRECTORY: int = auto()


class File(Node):
    """Manages a file on the filesystem."""

    def __init__(
        self: File,
        path: PathLike,
        mode: int = 0o644,
        action: FileAction = FileAction.CREATE,
        filetype: FileType = FileType.FILE,
        contents: str | bytes = "",
        dependencies: set[Node] = set(),
    ) -> None:

        super().__init__(dependencies)

        self._path: Path = path if isinstance(path, Path) else Path(path)
        self._mode: int = mode
        self._action: FileAction = action
        self._filetype: FileType = filetype
        self._contents: str | bytes = contents

        if action == FileAction.REMOVE:
            if mode is not None:
                raise ValueError("Cannot pass mode when removing file!")

            if contents:
                raise ValueError("Cannot pass contents when removing file!")

    def __repr__(self: File) -> str:
        contents: str = self._contents_as_str

        return (
            f"<{self.__class__.__name__} "
            + f"path={self._path} mode={self._mode} action={self._action} "
            + f"contents={contents if len(contents) <=8 else contents[:8] + '[...]'}>"
        )

    @property
    def _key(self: File) -> Hashable:
        return tuple((self._path, self._mode, self._action, self._contents))

    @cached_property
    def _contents_as_str(self: File) -> str:
        return (
            self._contents.decode()
            if isinstance(self._contents, bytes)
            else self._contents
        )

    def _create_directory(self: File) -> None:
        if self._mode is None:
            raise ValueError("Need to pass mode when creating file!")

        if not self._path.parent.exists():
            self._path.parent.mkdir(0o755, True)

    def _remove_file(self: File) -> None:
        shutil.rmtree(self._path, True)

    def _append_file(self: File) -> None:
        with self._path.open("a") as f:
            f.write(self._contents_as_str)

    def _create_file(self: File) -> None:
        self._create_directory()

        with self._path.open("w") as p:
            p.write(str(self._contents))

        os.chmod(self._path, self._mode)

    def _raise_unsupported(self: File) -> NoReturn:
        raise ValueError(f"{self._action} for {self._filetype} is not supported!")

    def act(self: File) -> None:
        {
            (FileAction.CREATE, FileType.FILE): self._create_file,
            (FileAction.CREATE, FileType.DIRECTORY): self._create_directory,
            (FileAction.APPEND, FileType.FILE): self._append_file,
            (FileAction.REMOVE, FileType.FILE): self._remove_file,
            (FileAction.REMOVE, FileType.DIRECTORY): self._remove_file,
        }.get((self._action, self._filetype), self._raise_unsupported)() # type: ignore


class Symlink(Node):
    """Creates a symbolic link from the target to the source."""

    def __init__(
        self: Symlink, source: Path, target: Path, dependencies: set[Node] = set()
    ) -> None:
        super().__init__(dependencies)

        self._source: Path = source
        self._target: Path = target

    @property
    def _key(self: Symlink) -> Hashable:
        return self._source, self._target

    def __repr__(self: Symlink) -> str:
        return (
            f"<{self.__class__.__name__} "
            + f"source={self._source} target={self._target}"
        )

    def act(self: Symlink) -> None:
        if not self._target.parent.exists():
            self._target.parent.mkdir(0o755, True)

        if not self._target.exists():
            self._target.symlink_to(self._source)
            return

        if self._target.is_dir():
            shutil.rmtree(self._target)
            return

        if self._target.is_symlink():
            self._target.unlink()
            return


class GitRepo(Node):
    """Clones a git repository to the local filesystem."""

    SSH_COMMAND: str = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    def __init__(
        self: GitRepo,
        remote: str,
        local: PathLike | None = None,
        dependencies: set[Node] = set(),
    ) -> None:
        super().__init__(dependencies)

        self._remote: str = remote
        self._local: PathLike | None = local

    @property
    def _key(self: GitRepo) -> Hashable:
        return (self._remote, self._local)

    def __repr__(self: GitRepo) -> str:
        return (
            f"<{self.__class__.__name__} url={self._remote}"
            + f"{' ' + str(self._local) if self._local is not None else ''}>"
        )

    def act(self: GitRepo) -> None:
        cmd: list[str | Path] = ["git", "clone", self._remote]

        if self._local is not None:
            cmd.append(
                self._local if isinstance(self._local, str) else str(self._local)
            )

        # fmt: off
        process(Command(
            cmd, env={**{"GIT_SSH_COMMAND": self.SSH_COMMAND}, **os.environ},
        ))
        # fmt: on


class GPGKey(Node):
    def __init__(
        self: GPGKey,
        keydata: BytesOrTextIO,
        dependencies: set[Node] = set(),
    ) -> None:
        super().__init__(dependencies)
        self._keydata: BytesOrTextIO = keydata

    @property
    def _key(self: GPGKey) -> Hashable:
        return self._keydata

    def act(self: GPGKey) -> None:
        process(Command(["gpg", "--import"], self._keydata))


def parse_args(args: list[str]) -> Namespace:
    parser: ArgumentParser = ArgumentParser(PROGNAME)
    parser.add_argument("-l", "--log-level", help="Log level", default="info")
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="Do not effect the system"
    )

    return parser.parse_args(args)


def dock_defaults(*defaults: tuple[str, MacOSDefaultValue]) -> list[MacOSDefault]:
    return [
        MacOSDefault("com.apple.dock", default[0], default[1]) for default in defaults
    ]


def main(argv: list[str]) -> int:
    """Main method for the module."""
    if not any(re.match(r"^SUDO_[^ ]+$", k) for k in os.environ):
        raise RuntimeError(f"{PROGNAME} is not running via sudo!")

    args: Namespace = parse_args(argv)
    logging.basicConfig(
        format=LOG_FMT,
        level=getattr(logging, args.log_level.upper()),
    )
    original_uid: int = int(cast(str, os.environ.get("SUDO_UID")))
    original_user: str = os.environ.get("SUDO_USER", "")

    # fmt: off
    # Needs to be done first and as root
    process(
        Serial([
            File(
                "/etc/sudoers.d/admin",
                0o644,
                contents="%admin ALL = (ALL) NOPASSWD : ALL\n"
            ),
            Command([
                "dseditgroup",
                "-o", "edit",
                "-a", original_user,
                "-t", "user",
                "admin",
            ]),
            Command(["chown", "-R", f"{original_user}:admin", "/opt"]),
        ]),
        *[
            MacOSFileFlag(MacOSFileFlagAction.CLEAR, MacOSFileFlagType.HIDDEN, cast(Path, path))
            for path in [Path("/Users") / original_user / "Library", "/Volumes"]
        ],
    )

    # Everything that follows is user-specific so switch back to original uid
    os.setuid(original_uid)
    home: Path = Path().home()
    os.chdir(home)
    base: Path = home / ".local" / "share" / PROGNAME
    files: Path = base / "files"
    icloud_docs: Path = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    os.environ["PATH"] = ":".join(["/opt/homebrew/bin", *os.environ["PATH"].split(":")])
    

    process(
        Hostname("rocinante"),
        Serial([
            Container({*dock_defaults(
                ("autohide", True),
                ("autohide-delay", 0.0),
                ("autohide-time-modifier", 0.0),
                ("expose-animation-duration", 0.1),
                ("minimize-to-application", True),
                ("show-recents", False),
                ("tilesize", 50),
            )}),
            Command(["killall", "Dock"])
        ]),
        MacOSDefault("com.apple.finder", "FXPreferredViewStyle", "Nlsv"),
        Serial([
            # Symlink(
            #     icloud_docs / "Secrets" / "personal.pem",
            #     home / ".ssh" / "personal.pem",
            #     {File(home / ".ssh", 0o755, filetype=FileType.DIRECTORY)},
            # ),
            GitRepo(f"git@github.com:gkze/{PROGNAME}", base),
            Container({
                Symlink(files / "alacritty.yml", home / ".config" / "alacritty" / "alacritty.yml"),
                Symlink(files / "git", home / ".config" / "git"),
                Symlink(files / "gitconfig", home / ".gitconfig"),
                Symlink(files / "nvim", home / ".config" / "nvim"),
                Symlink(files / "ssh_config", home / ".ssh" / "config"),
                Symlink(files / "starship.toml", home / ".config" / "starship.toml"),
                Symlink(files / "tmux.conf", home / ".config" / "tmux" / "tmux.conf"),
                Symlink(files / "topgrade.toml", home / ".config" / "topgrade.toml"),
                Symlink(files / "zshenv", home / ".zshenv"),
                Symlink(files / "zshrc", home / ".zshrc"),
            }),
        ]),
        Serial([
            GitRepo("https://github.com/Homebrew/brew", Path("/opt/homebrew")),
            Symlink(files / "Brewfile", home / "Brewfile"),
            Command(["brew", "bundle"]),
            GPGKey(StringIO((icloud_docs / "Secrets" / "george.kontridze@gmail.com.pgp").open("r").read())),
            Symlink(files / "sheldon_plugins.toml", home / ".sheldon" / "plugins.toml"),
            Command(["sheldon", "lock"])
        ]),
        GitRepo("git@github.com:tmux-plugins/tpm", home / ".config" / "tmux" / "plugins" / "tpm"),
        GitRepo(
            "git@github.com:wbthomason/packer.nvim",
            home / ".local" / "share" / "nvim" / "site" / "pack" / "packer" / "start" / "packer.nvim",
        )
    )
    # fmt: on

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
