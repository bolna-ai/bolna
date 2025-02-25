import asyncio
import inspect
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

class ObservableVariable:
    def __init__(self, value):
        self._value = value
        self._observers = []

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
                    # If an event loop is already running, schedule the async observer
                    loop = asyncio.get_running_loop()
                    loop.create_task(observer(new_value))
                except RuntimeError:
                    # No running loop; run the async function in a temporary event loop
                    asyncio.run(observer(new_value))
            else:
                # Synchronous observer: call it directly
                observer(new_value)