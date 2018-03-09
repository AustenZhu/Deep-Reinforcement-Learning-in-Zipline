from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import types

Base = declarative_base()

class JsonEncodedDict(types.TypeDecorator):
    impl = db.Text
    def process_bind_param(self, value, dialect):
        if value is None:
            return '{}'
        else: 
            return json.dumps(value)
    def process_result_value(self, value, dialect):
        if value is None:
            return {}
        else:
            return json.loads(value)
mutable.MutableDict.associate_with(JsonEncodedDict)

class Fundamentals(Base):
    __tablename__ = 'fundamentals'

    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    metric = Column(String)
    time_series = Column(JsonEncodedDict)

class Market_Metric(Base):
    __tablename__ = 'market_metrics'
    id = Column(Integer, primary_key=True)
    metric = Column(String)
    time_series = Column(JsonEncodedDict)