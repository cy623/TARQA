from datetime import datetime

def dt(value):
    return datetime.fromisoformat(str(value))

def stamp(value):
    return dt(value).isoformat(sep=" ", timespec="seconds")

def quote(value):
    return "'" + str(value).replace("'", "''") + "'"
