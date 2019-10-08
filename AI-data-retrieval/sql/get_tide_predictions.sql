/* Get tide prediction data for a specified station */
set nocount on
go

declare @NOW_START datetime
declare @NOW_END datetime
declare @STATION_ID varchar(7)
declare @DCP varchar(1)

/* Modify parameters here */
select @NOW_START="2005/01/01 00:00"
select @NOW_END="2005/12/31 23:59"
--select @NOW_END="2017/12/31 23:59"
select @STATION_ID = "8454000" -- Boston

select wa.STATION_ID, 
    wa.DATE_TIME,
    wasd.MSL,
    wa.WL_VALUE as "PRED_WL_VALUE",
    wa.WL_VALUE-wasd.MSL as "PRED_WL_VALUE_MSL"
from 
    OceanData.dbo.WL_PREDICTIONS wa,
    OceanData.dbo.WL_ACCEPTED_STATION_DATUM wasd
where
    wa.STATION_ID = @STATION_ID and   
    wa.DATE_TIME >= @NOW_START  and
    wa.DATE_TIME <= @NOW_END and
    wasd.STATION_ID = @STATION_ID and
    wasd.ACCEPTED_DATE_TIME is not null and
    wasd.VERIFIED_DATE_TIME is not null and
    wasd.EPOCH = "1983-2001" and
    wa.DATE_TIME between wasd.ACCEPTED_DATE_TIME and isnull(wasd.SUPERSEDED_DATE_TIME,getdate())
order by wa.DATE_TIME DESC    
