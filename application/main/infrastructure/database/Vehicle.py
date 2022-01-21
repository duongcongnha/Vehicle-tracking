from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Interval
from sqlalchemy.ext.declarative import declarative_base
from infrastructure.database.base import Base

class Vehicle(Base):
    __tablename__ = 'vehicle'
    # Here we define columns for the table vehicle
    # Notice that each column is also a normal Python instance attribute.
    ID = Column(String, primary_key=True)
    in_time = Column(DateTime)
    exit_time = Column(DateTime)
    type_vehicle = Column(String)
    lane = Column(String)
    
    def __init__(self, ID, in_time, exit_time, type_vehicle, lane):
        self.ID = ID
        self.in_time = in_time
        self.exit_time = exit_time
        self.type_vehicle = type_vehicle
        self.lane = lane

    def to_dict(self):
        FMT = "%d-%m-%Y %H:%M:%S"
        vehicle_dict = {}
        vehicle_dict[self.ID] = {}
        vehicle_dict[self.ID]['in_time'] = self.in_time.strftime(FMT)
        vehicle_dict[self.ID]['exit_time'] = self.exit_time.strftime(FMT)
        vehicle_dict[self.ID]['type_vehicle'] = self.type_vehicle
        vehicle_dict[self.ID]['lane'] = self.lane
        return vehicle_dict
