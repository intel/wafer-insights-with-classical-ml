from datetime import datetime, timedelta
import logging
import os
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import sqlite3
from Exceptions.exceptions import FailedConnToDB, MetaTableNotCreated, MetadataFailedInsertLoad, NoLoadData
from loaders.base_loader import utils, utils_schema
import yaml
import random
import numpy as np

if os.name == 'nt':
    import PyUber

    use_db2 = False
else:
    import pyodbc

    use_db2 = True


def get_project_root():
    file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
    os.path.dirname(file)
    return os.path.dirname(file)


class BaseLoader(object):
    def __init__(self, output_dir, source, sql_path, partition_cols, sanitize_db=True, recreate_db_from_cache=False,
                 backload=90, incload=12, write_parquet=False, custom_end_date=None, user_config_dir=None, lots=False):
        self._source = source
        self._sql_path = sql_path
        self._sql_filename = Path(sql_path).name
        self.output_dir = output_dir
        self.cache_dir = self.cache_path(output_dir)
        self._logger = self.setup_logger()
        self._sql = self.load_sql()
        self._backload = backload
        self._incload = timedelta(hours=incload)
        self._historypath = self.load_history_db()
        self._history_conn = None
        self._sanitize_db = sanitize_db
        self._recreate_db_from_cache = recreate_db_from_cache
        self._mclient = MongoClient()
        self._mdb = self._mclient
        self._partition_cols = partition_cols
        self._custom_end_date = custom_end_date
        self._write_parquet = write_parquet
        self._custom_lots = lots

        if use_db2:
            if user_config_dir is None:
                user_config_dir = str(Path(__file__).parent) + "/db_config/"
            self.db_config = yaml.load(open(str(Path(__file__).parent) + "/db_config/db_mapping.yaml"),
                                       Loader=yaml.FullLoader)
            self.user_config = yaml.load(open(user_config_dir + "user.yaml"), Loader=yaml.FullLoader)

    def setup_logger(self):
        logger = logging.getLogger(__name__ + self._source[0])
        if self.cache_dir != '':
            p = Path(self.cache_dir + "/Logs")
        if not p.exists():
            p.mkdir(parents=True)

        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S', filename=str(p) + "/" + self._source[0], filemode='a')
        logging.getLogger('cassandra').setLevel(logging.INFO)
        if not use_db2:
            logging.getLogger('PyUber').setLevel(logging.CRITICAL)

        return logger

    # scylla
    def create_table(self, conn):
        sql = '''CREATE TABLE IF NOT EXISTS LOAD_HISTORY (
            END_LOAD_DATE timestamp ,
            START_LOAD_DATE timestamp,
            REAL_LOAD_DATE timestamp,
            SOURCE text,
            PRIMARY KEY (END_LOAD_DATE, SOURCE));'''
        try:
            curs = conn.cursor()
            curs.execute(sql)
            conn.commit()
        except:
            self._logger.exception(MetaTableNotCreated.msg)

    def insert_load(self, param_tuple):
        sqlite_insert_with_param = """INSERT INTO 'LOAD_HISTORY'
                                  ('END_LOAD_DATE', 'START_LOAD_DATE', 'REAL_LOAD_DATE', 'SOURCE') 
                                  VALUES (?, ?, ?, ?);"""
        self._logger.info(f'Inserting into metadata values : {param_tuple}')
        cursor = self._history_conn.cursor()
        try:
            cursor.execute(sqlite_insert_with_param, param_tuple)
            self._history_conn.commit()
        except:
            self._logger.exception(MetadataFailedInsertLoad.msg)

    def get_last_load(self, source):
        default_start_time = datetime.now() - timedelta(days=self._backload)
        conn = sqlite3.connect(self._historypath,
                               detect_types=sqlite3.PARSE_DECLTYPES |
                                            sqlite3.PARSE_COLNAMES)
        self._history_conn = conn
        self.create_table(conn)
        if self._recreate_db_from_cache:
            self._logger.info(f'rebasing our metadata from cache from source: {source}.')
            db_cache_time = self.db_rebase_from_cache(source)
            if default_start_time < db_cache_time:
                self._logger.info('Cache has a later date than default start time, using latest cache time.')
                return db_cache_time
            else:
                self._logger.info('Default start time will be used.')
                return default_start_time

        data = pd.read_sql("select REAL_LOAD_DATE FROM LOAD_HISTORY WHERE SOURCE=?", conn,
                           params=(str(source),))
        if data.shape[0] == 0:
            self._logger.info('data shape is 0, using default start time.')
            return default_start_time

        try:
            real_load_date = data["REAL_LOAD_DATE"].max().to_pydatetime()
            self._logger.info(f'Getting last load date:{real_load_date}')
            return real_load_date
        except:
            self._logger.exception(NoLoadData.msg)

    @utils.retry(Exception, tries=4)
    def load_from_db(self, conn, start_time, end_time, source):
        if not self._custom_lots:
            try:
                if not use_db2:
                    cursor = conn.execute(self._sql, START=start_time, END=end_time)
                else:
                    sql = self._sql.replace(":START", '?').replace(":END", "?")
                    cursor = conn.execute(sql, (start_time, end_time))
            except:
                self._logger.exception(FailedConnToDB.msg)

            columns = [x[0] for x in cursor.description]
            df = pd.DataFrame().from_records(cursor.fetchall(), columns=columns)
            if df.empty:
                self._logger.info(f'Dataframe is empty. No data between time: {start_time} - {end_time}')
                start_time = end_time
                end_time = end_time + self._incload
                return start_time, end_time
        if self._sql_filename == 'borissortdiagdistinct.sql' or self._sql_filename == 'borissortinddistinct.sql' or self._sql_filename == 'processrevdistinct.sql':
            if self._custom_lots:
                lots = self._custom_lots
            else:
                lots = df['LOT7'].unique().tolist()
            real_load_time = self.split_boris_into_lots(lots, source, start_time, end_time, conn)
            if not real_load_time:
                self._logger.info(f'Dataframe is empty. No data between time: {start_time} - {end_time}')
                start_time = end_time
                end_time = end_time + self._incload
                return start_time, end_time
        else:

            self._logger.info(f"{len(df['LOT7'].unique())}")
            if not use_db2:
                schema = utils_schema.getSchema(cursor)
            else:
                schema = utils_schema.getSchemaODBC(cursor)

            df = utils_schema.standardizeSchema(schema, df)
            df.replace("None", "", inplace=True)
            df['PROCESS'] = df['PROCESS'].astype(str).str.slice(stop=4)
            self.insert_data_to_db(df)
            real_load_time = df['LOAD_DATE'].max().to_pydatetime()

            if self._write_parquet:
                self._logger.info('Writing to parquet.')
                t = pa.Table.from_pandas(df)
                pq.write_to_dataset(t, self.cache_dir, partition_cols=self._partition_cols, flavor='spark')
        self.insert_load((end_time, start_time, real_load_time, source))
        self._logger.info(f'Load from db between times {start_time} - {end_time} complete.')
        start_time = end_time
        end_time = end_time + self._incload
        return start_time, end_time

    def split_boris_into_lots(self, lots, source, start_time, end_time, conn):
        if self._sql_filename == 'borissortdiagdistinct.sql':
            boris_sql = Path(str(Path(__file__).parent.parent) + "/boris_sort_diag/sql/borissortdiag.sql").read_text()
        elif self._sql_filename == 'borissortinddistinct.sql':
            boris_sql = Path(
                str(Path(__file__).parent.parent) + "/boris_sort_indicator/sql/borissortind.sql").read_text()
        elif self._sql_filename == 'processrevdistinct.sql':
            boris_sql = Path(str(Path(__file__).parent.parent) + "/process_rev/sql/process_rev.sql").read_text()
        site = source[:3]
        process = source[-4:]
        boris_sql = boris_sql.replace("{SITE}", site).replace(
            "{TECHNOLOGY}", process)

        self._logger.info(f"amount of lots: {len(lots)}")
        chunk_size = 10
        lots = [lots[i:i + chunk_size] for i in range(0, len(lots), chunk_size)]
        real_load_time = ''
        for chunk in lots:
            copy_boris_sql = boris_sql
            self._logger.info(f'Load len chunk: {len(chunk)}')
            chunkstr = ",".join("'{0}'".format(lot) for lot in chunk)
            copy_boris_sql = copy_boris_sql.replace("{LOTS}", chunkstr)
            cursor = conn.execute(copy_boris_sql, START=start_time, END=end_time)
            columns = [x[0] for x in cursor.description]
            df = pd.DataFrame().from_records(cursor.fetchall(), columns=columns)
            if df.empty:
                self._logger.info('empty')
                continue
            if not use_db2:
                schema = utils_schema.getSchema(cursor)
            else:
                schema = utils_schema.getSchemaODBC(cursor)

            df = utils_schema.standardizeSchema(schema, df)
            df.replace("None", "", inplace=True)
            df['PROCESS'] = df['PROCESS'].astype(str).str.slice(stop=4)
            self.insert_data_to_db(df)

            real_load_time = df['LOAD_DATE'].max().to_pydatetime()
            if self._write_parquet:
                self._logger.info('Writing to parquet.')
                t = pa.Table.from_pandas(df)
                pq.write_to_dataset(t, self.cache_dir, partition_cols=self._partition_cols, flavor='spark')
        if not real_load_time:
            return False
        return real_load_time

    def parse_to_mongo(self, group, columns, values, df, mdb):
        startm = datetime.now()
        self._logger.info(f'Writing bulk of {columns} to mongo.')
        if isinstance(df, pd.DataFrame):
            gb = df.groupby(group)
        else:
            gb = df

        if len(gb) == 0:
            return

        ops = []
        for index, ret in gb:
            lot = index[0]
            wafer = index[1]
            record = dict(zip(ret[columns], ret[values]))

            new = {}
            for key, val in record.items():
                if not pd.isnull(val):
                    if not val == "":
                        new[key] = val

            if len(new) == 0:
                continue

            for idx in range(len(group)):
                new[group[idx]] = index[idx]

            ops.append(UpdateOne({'LOT7': lot, 'WAFER3': wafer}, {'$set': new}, upsert=True))

            if len(ops) == 1000:
                try:
                    mdb.bulk_write(ops, ordered=False)
                    ops = []
                except BulkWriteError as bwe:
                    self._logger.exception("Error encountered in Baseloader bulk load to mongo.")
                    self._logger.exception(bwe.details)
                    # you can also take this component and do more analysis
                    # werrors = bwe.details['writeErrors']
                    raise
        if len(ops) > 0:
            try:
                mdb.bulk_write(ops, ordered=False)
            except BulkWriteError as bwe:
                self._logger.exception("Error encountered in Baseloader bulk load to mongo.")
                self._logger.exception(bwe.details)
                # you can also take this component and do more analysis
                # werrors = bwe.details['writeErrors']
                raise

        endm = datetime.now()
        self._logger.info(f"Took {(endm - startm).total_seconds()} to load bulk of {columns} to mongodb")

    def parse_blobs_to_mongo(self, groups, gb, name_col, blob_col, mdb):
        ops = []
        for index, df in gb:
            idx_dict = dict(zip(groups, index))

            wafermaps = df[blob_col].map(self.decode)
            names = df[name_col]
            zpd = zip(names, wafermaps)
            for name, wm in zpd:
                lot = index[0]
                wafer = index[1]
                spec_dict = dict(idx_dict)
                spec_dict[name_col] = name
                spec_dict['WAFERMAP'] = wm

                if spec_dict['WAFERMAP'] is None:
                    continue

                ops.append(UpdateOne({'LOT7': lot, 'WAFER3': wafer}, {'$set': spec_dict}, upsert=True))
                if len(ops) == 1000:
                    try:
                        mdb.bulk_write(ops, ordered=False)
                        ops = []
                    except BulkWriteError as bwe:
                        self._logger.exception("Error encountered in Baseloader bulk load to mongo.")
                        self._logger.exception(bwe.details)
                        # you can also take this component and do more analysis
                        # werrors = bwe.details['writeErrors']
                        raise
        if len(ops) > 0:
            try:
                mdb.bulk_write(ops, ordered=False)
                ops = []
            except BulkWriteError as bwe:
                self._logger.exception("Error encountered in Baseloader bulk load to mongo.")
                self._logger.exception(bwe.details)
                # you can also take this component and do more analysis
                # werrors = bwe.details['writeErrors']
                raise

    def get_conn(self, datasource):
        if not use_db2:
            return PyUber.connect(datasource=datasource)

        usr = self.user_config['USER']
        pwd = self.user_config['PASSWORD']

        connstring = self.db_config[datasource]
        connstring = connstring + f";UID={usr};PWD={pwd}"
        conn = None

        try:
            conn = pyodbc.connect(connstring)
        except pyodbc.Error as err:
            print(err)

        return pyodbc.connect(connstring)

    def load_to_cache(self):
        for source in self._source:
            self._logger.info(f"Querying {source}")
            start_time = self.get_last_load(source)
            # loading 12h start time - data change specifically for boris but helpful to consolidate other domains.
            minutes = random.randint(0, 30)
            start_time = start_time - timedelta(hours=12, minutes=minutes)
            self._logger.info(f"starting to load data from: {start_time}")
            end_time = start_time + self._incload
            try:
                # conn = PyUber.connect(datasource=source)
                conn = self.get_conn(datasource=source)
            except:
                self._logger.exception(FailedConnToDB.msg)

            if self._sql_filename == 'borissortdiag' or 'borissortind' or 'process_rev.sql':
                site = source[:3]
                process = source[-4:]
                self._sql = self._sql.replace("{SITE}", site).replace(
                    "{TECHNOLOGY}", process)
            if self._custom_end_date is None:
                finish_time = datetime.now() - timedelta(hours=1)
            else:
                finish_time = self._custom_end_date
            while start_time < finish_time:

                if end_time > finish_time:
                    end_time = finish_time

                self._logger.info(f"{source}: {start_time} -- {end_time}")
                try:
                    start_time, end_time = self.load_from_db(conn, start_time,
                                                             end_time, source)
                except:
                    self._logger.exception('An error occured in load_from_db')
            self._logger.info("Loading data for session complete")
            if self._sanitize_db:
                self.db_delete_old_data(source)

    def mongo_reload_from_cache(self, path, type):
        pds = pq.ParquetDataset(path)
        for par in pds.pieces:
            df = par.read().to_pandas()
            self.insert_data_to_db(df)

    def db_rebase_from_cache(self, source, filepath):
        if filepath:
            t = pq.ParquetDataset(filepath).read(columns=['TEST_END_DATE'])
            df = t.to_pandas()
            end_load_date = df['TEST_END_DATE'].max().to_pydatetime()
            start_load_date = df['TEST_END_DATE'].min().to_pydatetime()
            sql_text = f"INSERT INTO 'LOAD_HISTORY' (END_LOAD_DATE," \
                       f" START_LOAD_DATE, REAL_LOAD_DATE, SOURCE)" \
                       f" VALUES (?, ?, ?, ?);"
            params = (end_load_date, start_load_date, end_load_date, source)
            cursor = self._history_conn.cursor()
            cursor.execute(sql_text, params)
            self._history_conn.commit()
            return end_load_date

    def db_delete_old_data(self, source):
        last_load = self.get_last_load(source)
        sql_del = f"DELETE FROM LOAD_HISTORY WHERE END_LOAD_DATE > '{last_load}" \
                  f"' AND SOURCE= '{source}'"
        self._history_conn.execute(sql_del)
        self._history_conn.commit()

    def load_history_db(self):
        dir_path = self.cache_dir + r"/sqlite/history.db"
        p = Path(self.cache_dir + r"/sqlite")

        if not p.exists():
            p.mkdir(parents=True)

        self._logger.info(f'metadata directory: {dir_path}')
        return dir_path

    def load_sql(self):
        self._logger.info(f'Load sql from: {self._sql_path}')
        return Path(self._sql_path).read_text()

    def cache_path(self, outputdir):
        if outputdir == '':
            p = Path(__file__).parents[4]
            cache_path = str(p) + f'\caches\{self._sql_filename}'
            return cache_path
        else:
            p = Path(outputdir + f"\{type(self).__name__}")
            if not p.exists():
                p.mkdir(parents=True)

            return str(p)

    def check_mongo_index(self, db, indexes, unique=False):
        for index in indexes:
            db.create_index(index, unique=unique)

    @utils.retry(Exception, tries=10, backoff=1, delay=5)
    def insert_to_cassandra(self, db, cql, vals):
        db.execute(cql, vals)

    def cassandra_keyspace_exist(self, mc, collection_name):
        session = mc.connect()
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {collection_name} WITH REPLICATION ="
            " { 'class' : 'SimpleStrategy', 'replication_factor' : 2 }")

    '''
    Below here, must override with child functions.
    '''

    def insert_data_to_db(self, df):
        pass

# Please run child loader for specific data you want to load.
