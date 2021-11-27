# syscfg: System Configuration

A lightweight system configurator with a declarative API.
Nice:
* Only requires Python3
* Dependency-free, single-file (`curl ... | sudo python3 ...`)

Not very nice:
* Currently only supports macOS
* Is specific to me (George)

example:

```python
...[directly in single file so logic code above]...
process(
  Command(["echo", "hello", "world"]),
  Serial([
    Command(["echo", "this", "first"]),
    Command(["echo", "this", "second"]),
  ])
)
...[graph processing code below]...
```

At its core, syscfg is a concurrent task graph processor. Each node is
customized to a speific task via a subclass. Here is the simplest task - a
command:

```python
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
```

## LICENSE

[MIT](LICENSE)
