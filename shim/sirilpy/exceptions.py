class SirilError(Exception):
    """Γενικό σφάλμα (συμβατότητα με το sirilpy)."""


class SirilConnectionError(SirilError):
    pass


class NoImageError(SirilError):
    pass
