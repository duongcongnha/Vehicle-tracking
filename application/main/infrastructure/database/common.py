from infrastructure.database.base import session_factory
from infrastructure.database.Vehicle import Vehicle
from sqlalchemy import and_
from datetime import datetime
from typing import List

def add_vehicle_to_db(vehicle:Vehicle):
    session = session_factory()
    session.add(vehicle)
    session.commit()
    session.close() 

def get_all_vehicles():
    session = session_factory()
    vehicles_query = session.query(Vehicle)
    session.close()
    return vehicles_query.all()

def list_vehicles_to_list_dict(all_vehicles:List[Vehicle]):
    result = [x.to_dict() for x in all_vehicles]
    return result

def get_vehicles_in_timerange(start_datetime:datetime, end_datetime:datetime):
    session = session_factory()
    vehicles_query = session.query(Vehicle).filter(and_(Vehicle.in_time >= start_datetime,\
                                                     Vehicle.in_time <= end_datetime))
    session.close()
    all_vehicles = vehicles_query.all()
    all_vehicles = list_vehicles_to_list_dict(all_vehicles)
    return all_vehicles
