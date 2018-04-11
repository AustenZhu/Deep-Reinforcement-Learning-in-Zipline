from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

#Replace with absolute path
engine = create_engine('sqlite:////home/austen/sp18/CIB/dynamic_beta_research/zip_dl_infra/zipdl/data/data.db')
def create_session(autoflush=True):
    Session = sessionmaker(bind=engine,autoflush=autoflush)
    session = Session()
    return session