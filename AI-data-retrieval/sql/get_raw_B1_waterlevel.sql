/* Get raw B1 backup data for DCP 1 data for a specified station */
set nocount on
go

declare @NOW_START datetime
declare @NOW_END datetime
declare @STATION_ID varchar(7)
declare @DCP varchar(1)

/* Modify parameters here */
select @NOW_START="2006/01/01 00:00"
select @NOW_START
--select @NOW_END="2006/06/01 23:59"
select @NOW_END="2017/12/31 23:59"
select @STATION_ID = "8536110" 
select @DCP = "1"


select d.STATION_ID, 
    wa.DATE_TIME,
    wasd.MSL,
    (wa.WL_VALUE*convert(float,sp1.PARAMETER_VALUE)  + convert(float,sp.PARAMETER_VALUE)) as "WL_VALUE"  ,
    (wa.WL_VALUE*convert(float,sp1.PARAMETER_VALUE)  + convert(float,sp.PARAMETER_VALUE))-wasd.MSL as "WL_VALUE_MSL",
    wa.WL_SIGMA,
    sp.PARAMETER_VALUE as "ACC_BACKUP_OFFSET",
    sp1.PARAMETER_VALUE as "ACC_BACKUP_GAIN"
from  OceanData.dbo.WL_BACKUP wa
        join OceanMD.dbo.DEPLOYMENT d on wa.DEPLOYMENT_ID = d.DEPLOYMENT_ID 
        join OceanData.dbo.WL_ACCEPTED_STATION_DATUM wasd on d.STATION_ID=wasd.STATION_ID and wasd.ACCEPTED_DATE_TIME is  not null and
                                                                                  wasd.VERIFIED_DATE_TIME is not null and wasd.EPOCH = "1983-2001"
        join OceanMD.dbo.SENSOR_PARAMETER sp on sp.STATION_ID=d.STATION_ID and sp.DCP=d.DCP and sp.SENSOR_ID=d.SENSOR_ID
        join OceanMD.dbo.PARAMETER p on p.PARAMETER_ID=sp.PARAMETER_ID
        join OceanMD.dbo.SENSOR_PARAMETER sp1 on sp1.STATION_ID=d.STATION_ID and sp1.DCP=d.DCP and sp1.SENSOR_ID=d.SENSOR_ID
        join OceanMD.dbo.PARAMETER p1 on p1.PARAMETER_ID=sp1.PARAMETER_ID        
where
    d.STATION_ID = @STATION_ID and   
    d.DCP = @DCP and
    wa.DATE_TIME between @NOW_START and @NOW_END and
    wa.DATE_TIME between d.DEPLOY_DATE_TIME and isnull(d.REMOVE_DATE_TIME, getdate()) and
    wa.DATE_TIME between sp.BEGIN_DATE_TIME and isnull(sp.END_DATE_TIME, getdate()) and
    wa.DATE_TIME between sp1.BEGIN_DATE_TIME and isnull(sp1.END_DATE_TIME, getdate()) and
    p1.PARAMETER_NAME='ACC_BACKUP_GAIN' and
    p.PARAMETER_NAME = 'ACC_BACKUP_OFFSET' and
    wa.DATE_TIME between wasd.ACCEPTED_DATE_TIME and isnull(wasd.SUPERSEDED_DATE_TIME,getdate())
order by wa.DATE_TIME DESC   


