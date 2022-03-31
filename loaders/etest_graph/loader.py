
from connectors.database import get_connection,


sql = """
SELECT /*+  ordered index(ett et_tp_str_uk) index(ets wesr_pk) */
          --(SELECT pl99.PROCESS_SECURED_BY FROM A_LOT_AT_OPERATION pl99 WHERE ets.lot = pl99.lot AND ets.operation = pl99.operation AND ets.facility = pl99.facility AND rownum <= 1) AS PROCESS
          SUBSTR(ets.PROCESS,2,4) as PROCESS
         ,ets.LOAD_END_DATE_TIME as LOAD_DATE
         ,ets.facility AS facility
         ,SUBSTR(ets.lot,1,7) AS LOT7
         ,ets.devrevstep AS DEVREVSTEP
         ,substr(ets.devrevstep,1,4) as SHORTDEVICE
         ,ets.wafer_id AS WAFER3
         ,ets.wafer_scribe AS WAFER
         ,ets.operation AS OPERATION
         ,ets.test_end_date_time AS TEST_END_DATE
         ,ets.program_name AS PROGRAM_NAME
         ,ett.test_name AS TEST_NAME
         ,ett.structure_name AS structure_name
         ,wer.median AS median
         ,wer.mean AS mean
         ,wer.stdev AS stdev
FROM
A_ETest_Testing_Session ets
INNER JOIN A_ETest_Test ett ON ets.program_name = ett.program_name
INNER JOIN A_Wafer_Etest_Setup_Rollup wer ON ets.lao_start_ww = wer.lao_start_ww AND ets.les_id = wer.les_id AND ets.wafer_id = wer.wafer_id AND ett.Test_Name = wer.Test_Name AND ett.Structure_Name = wer.Structure_Name
WHERE
              ets.valid_flag = 'Y'
AND  ets.LOAD_END_DATE_TIME > :START AND ets.LOAD_END_DATE_TIME <= :END AND ett.structure_name NOT LIKE '%PROBE%' AND ett.structure_name NOT LIKE '%RALPH%' AND ett.structure_name NOT LIKE '%PRBRES%' and ett.structure_name NOT LIKE '%LISA%'
AND SUBSTR(ets.PROCESS,2,4) NOT IN ('1276', '1278')
"""

def query_chunk(connstring, query, start, end):

    raise NotImplemented()

def update_cache():
    raise NotImplemented()

