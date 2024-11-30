from pydantic import BaseModel

class ErrorModel(BaseModel):
    """Modelo para respostas de erros
    """
    mensagem: str
    exception: str

    @classmethod
    def from_exception(cls, exception: Exception):
        return cls(
            mensagem=str(exception),
            exception=exception.__class__.__name__
        )
