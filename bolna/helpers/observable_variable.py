import asyncio
import inspect
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ObservableVariable:
    def __init__(self, value):
        self._value = value
        self._observers = []
        # Strong references to in-flight async-observer tasks. asyncio only keeps a weak
        # reference to a bare task, so without this the event loop may garbage-collect an
        # observer coroutine before it finishes — silently dropping lifecycle events such as
        # agent hangup, end-of-call teardown, and web-call init.
        self._pending_tasks = set()

    def add_observer(self, observer):
        """
        Register an observer function.
        The observer can be a synchronous function or an async function.
        """
        self._observers.append(observer)

    @property
    def value(self):
        """Getter for the observable variable."""
        return self._value

    @value.setter
    def value(self, new_value):
        """Setter that updates the variable and notifies observers if the value changes."""
        if self._value != new_value:
            self._value = new_value
            self._notify_observers(new_value)

    def _notify_observers(self, new_value):
        """
        Notify each observer about the new value.
        Async observers are scheduled appropriately.
        """
        for observer in self._observers:
            if inspect.iscoroutinefunction(observer):
                try:
                    # If an event loop is already running, schedule the async observer and
                    # keep a strong reference to the task until it completes (see _pending_tasks).
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(observer(new_value))
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)
                except RuntimeError:
                    # No running loop; run the async function in a temporary event loop
                    asyncio.run(observer(new_value))
            else:
                # Synchronous observer: call it directly
                observer(new_value)
