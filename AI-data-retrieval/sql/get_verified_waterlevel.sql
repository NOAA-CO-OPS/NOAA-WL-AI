/* Get verified water level data for DCP 1 data for a specified station */
set nocount on
go

declare @NOW_START datetime
declare @NOW_END datetime
declare @STATION_ID varchar(7)
declare @DCP varchar(1)

/* Modify parameters here */
select @NOW_START="2001/01/01 00:00"
select @NOW_END="2001/01/01 23:59"
--select @NOW_END="2017/12/31 23:59"
select @STATION_ID = "8443970" -- Boston

select wa.STATION_ID, 
    wa.DATE_TIME,
    wasd.MSL,
    wa.WL_VALUE as "VER_WL_VALUE",
    wa.WL_VALUE-wasd.MSL as "VER_WL_VALUE_MSL",
    wa.WL_SIGMA as "VER_WL_SIGMA"
from 
    OceanData.dbo.WL_ACCEPTED_6MIN wa,
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
