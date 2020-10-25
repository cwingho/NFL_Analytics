from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from plotnine import *
#pip install plotly==3.10.0
#pip install plotly==4.1.0
#import plotly.plotly as py
#import plotly.graph_objs as go

#pip install chart_studio
#import chart_studio.plotly as py
#import plotly.graph_objects as go

import matplotlib.pyplot as plt
import pandas

if __name__ == "__main__":

    spark = SparkSession.builder.appName('cs5488project').getOrCreate()
    injury = spark.read.option("delimiter", ",").csv("PlayerTrackData.csv/InjuryRecord.csv",header=True,inferSchema=True)
    playlist = spark.read.option("delimiter", ",").csv("PlayerTrackData.csv/PlayList.csv",header=True,inferSchema=True)
    #playtrack = spark.read.option("delimiter", ",").csv("PlayerTrackData.csv/PlayerTrackData.csv",header=True,inferSchema=True)

    injury.printSchema()
    playlist.printSchema()
    #playtrack.printSchema()

    injury = injury.withColumn("Severity", col("DM_M1") + col("DM_M7") + col("DM_M28") + col("DM_M42"))
    injury = injury.withColumn("SevLevel", when(col("Severity") >=3, "H").otherwise("L"))
    injury = injury.select("PlayerKey","GameID","PlayKey","BodyPart","Surface","Severity","SevLevel")
 
    # 1, injury count by surface, severity -- a slightly higher rate for severe injury (missed day > 28 (DM_28 / DM_42)) for synthetic surface
    severity = injury.groupBy('Severity', 'Surface').count().orderBy(col("Severity").asc(), col('Surface').desc()).toPandas()
    sevLevel = injury.groupBy('SevLevel', 'Surface').count().orderBy(col("SevLevel").asc(), col('Surface').desc()).toPandas()
    
    p1 = ggplot(severity, aes(x='Severity', y='count')) + geom_line(aes(fill='Surface', color='Surface'), size=1.5) +\
        labs(x="Severity Least(1) to Most(4)", y='No. of Injury', title = 'Count of Injury by Severity')
    print(p1)
    
    p2 = ggplot(sevLevel, aes(x='SevLevel', y='count')) + geom_point(aes(color='Surface'), size=5) +\
        labs(x="Severity Level (L and H)", y='No. of Injury', title = 'Count of Injury by Severity Level')
    print(p2)    
    
    # 2, injury count by body part, severity, surface 
    # -- analyse by body part: ankle shows more severe injury on synthetic surface, knee and toes injury do not show significant difference on both surfaces
    # -- surprisingly, foot has more severe injury reported on natural surface  
    sevBodyPart = injury.groupBy('BodyPart', 'Severity','Surface').count().orderBy(col('BodyPart').asc(), col('Severity').desc(),col('Surface').asc()).toPandas()
    p3 = ggplot(sevBodyPart, aes(x='Severity', y="count")) + \
        geom_point(aes(color='BodyPart'), size=4) + \
        facet_grid('Surface ~ BodyPart') + \
        labs(x="Severity Least(1) to Most(4)", y='No. of Injury', title = 'Count of Injury by Severity')
    
    print(p3)
            

    # 3, injury count by roster position, severity, surface 
    # -- the percentage of record without roster position is significant in this dataset, the data are too scattered
    # -- no observation can be made
    detail = injury.join(playlist, injury.PlayKey == playlist.PlayKey, 'left')
    sevRosterPos = detail.groupBy('Surface','Severity','RosterPosition').count().orderBy(col('Severity').desc(), col('RosterPosition').asc()).toPandas()
    
    p4 = ggplot(sevRosterPos, aes(x='Severity', y="count")) + \
        geom_point(aes(color='RosterPosition'), size=4) + \
        facet_grid('Surface ~ ') + \
        labs(x="Severity Least(1) to Most(4)", y='No. of Injury', title = 'Count of Injury by Severity')
    print (p4)
    
    spark.stop()