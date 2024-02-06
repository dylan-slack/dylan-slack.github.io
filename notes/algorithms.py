from typing import Any

time_input = "March 5 1972 1:51 AM"

mon_2_days = [
    ("January", 31),
    ("February", 28),
    ("March", 31),
    ("April", 30),
    ("May", 31),
    ("June", 30),
    ("July", 31),
    ("August", 31),
    ("September", 30),
    ("October", 31),
    ("November", 30),
    ("December", 31)
]


class Time:
    """Defines a particular time object"""
    def __init__(self, time: int, data: Any):
        self.time = time
        self.data = data

class TimeStore:
    """Defines the timestore"""
    def __init__(self):
        self.store = []

    @staticmethod
    def _str_to_int(val: str) -> int:
        ss = val.split()
        seconds = 0
        year = int(ss[2])
        day = int(ss[1])
        seconds += year * 365 * 24 * 60 * 60
        seconds += day * 24 * 60 * 60
        days_from_month = 0
        for d in mon_2_days:
            if d[0] == ss[0]:
                break
            days_from_month += d[1]
        seconds += days_from_month * 24 * 60 * 60
        hour_min = ss[3].split(":")
        hour = int(hour_min[0])
        sec = int(hour_min[1]) * 60
        seconds += sec + (hour * 60 * 60)
        if ss[4] == "PM":
            seconds += (12 * 60 * 60)
        return seconds

    def store(self, val: str, data: Any):
        """Store"""
        seconds = self._str_to_int(val)
        nt = Time(seconds, data)
        self._insert(nt)

    def _insert(self, nt: Time):
        """Insertion"""
        i, j = 0, len(self.store) - 1
        while i <= j:
            cur = (i + j) // 2
            if self.store[cur].time <= nt.time:
                if cur < len(self.store) - 1 and self.store[cur+1].time > nt.time:
                    self.store.insert(nt, cur)
                else:
                    

ts = TimeStore()
print(ts._str_to_int(time_input))




