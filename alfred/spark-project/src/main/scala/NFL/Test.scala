package NFL

import org.apache.spark.sql.functions.{col, column, expr}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
//https://medium.com/expedia-group-tech/deep-dive-into-apache-spark-window-functions-7b4e39ad3c86
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.attribute._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import NFLSchema.dataSchema
import NFLModel.ModelProcessing

object Test extends App{

  Logger.getLogger("org.apache.spark").setLevel(Level.OFF)

  var injureyFilePath = "src/main/resources/data/injuryRecord.csv"
  var playListFilePath = "src/main/resources/data/PlayList.csv"
  var playTrackFilePath = "src/main/resources/data/PlayerTrackData.csv"

  var schemaObj = new dataSchema()
  //var injuryDF = schemaObj.readFile(injureyFilePath, schemaObj.injurySchema)
  //val newInjuryDF = schemaObj.constructInjuryDF(injuryDF)
  //numOfPlayerGamePlayDF.show(50)
/*
  val pivot_weatherDF = numOfPlayerGamePlayDF.filter(col("play_PlayerKey") === "33474")
    .select("play_PlayerKey","play_GameID","Weather")
    .groupBy("play_PlayerKey").pivot("Weather").count()
    .withColumnRenamed("play_PlayerKey","play_PlayerKey_w")

  val pivot_stadiumTypeDF = numOfPlayerGamePlayDF.filter(col("play_PlayerKey") === "33474")
    .select("play_PlayerKey","play_GameID","StadiumType")
    .groupBy("play_PlayerKey").pivot("StadiumType").count()
    .withColumnRenamed("play_PlayerKey","play_PlayerKey_s")

  val pivot_fieldTypeDF = numOfPlayerGamePlayDF.filter(col("play_PlayerKey") === "33474")
    .select("play_PlayerKey","play_GameID","FieldType")
    .groupBy("play_PlayerKey").pivot("FieldType").count()
    .withColumnRenamed("play_PlayerKey","play_PlayerKey_f")

  pivot_weatherDF.show()
  pivot_stadiumTypeDF.show()
  pivot_fieldTypeDF.show()

  val combineDF = pivot_fieldTypeDF.join(pivot_stadiumTypeDF,col("play_PlayerKey_f") === col("play_PlayerKey_s"),"inner")
    .join(pivot_weatherDF,col("play_PlayerKey_w") === col("play_PlayerKey_f"),"inner")

  combineDF.show()
*/
  //var playerTrackDF = schemaObj.readFile(playTrackFilePath, schemaObj.playerTrackSchema)

  //var finalPlayerTrackDF = schemaObj.constructPlayerTrackDF(playerTrackDF)

  //finalPlayerTrackDF.printSchema()

  //numOfPlayerGamePlayDF.write.mode("append").parquet("src/main/resources/data/numOfPlayerGamePlayDF.parquet")

 // val playListDF = schemaObj.readFile(playListFilePath,schemaObj.playListSchema)

 // var newplayListDF = schemaObj.constructPlayList(playListDF)
 // newplayListDF.printSchema()

 val playListDF = schemaObj.readFile(playListFilePath,schemaObj.playListSchema)
/*
  var newplayListDF = playListDF.na.fill("Missing",schemaObj.stringColListInPlayer)
    .withColumn("PlayType", when(col("PlayType") ==="0","Missing").otherwise(col("PlayType")))
    .withColumn("StadiumType", when(col("StadiumType").isin(schemaObj.dome_open:_*),"dome_open")
      .when(col("StadiumType").isin(schemaObj.dome_closed:_*),"dome_closed")
      .when(col("StadiumType").isin(schemaObj.indoor_open:_*),"indoor_open")
      .when(col("StadiumType").isin(schemaObj.indoor_closed:_*),"indoor_closed")
      .when(col("StadiumType").isin(schemaObj.outdoor:_*),"outdoor")
      .otherwise(col("StadiumType")))
    .withColumn("Weather", when(col("Weather").isin(schemaObj.overcast:_*),"overcast")
      .when(col("Weather").isin(schemaObj.clear:_*),"clear")
      .when(col("Weather").isin(schemaObj.rain:_*),"rain")
      .when(col("Weather").isin(schemaObj.snow:_*),"snow")
      .when(col("Weather").isin(schemaObj.weathermissing:_*),"Missing")
      .otherwise(col("Weather")))
    .withColumn("Temperature", when(col("Temperature") < 0, 61).otherwise(col("Temperature")))

  newplayListDF.printSchema()
  //newplayListDF.groupBy(col("Weather")).count().show()
  var gameListDF = newplayListDF.filter(col("play_PlayerKey") === "26624")
    .select(col("play_PlayerKey").as("GamePlayerID"), col("play_GameID").as("GameID"), col("PlayerDay"),
    col("PlayerGame"), col("RosterPosition"),col("StadiumType"),
    col("FieldType"),col("Temperature"),col("Weather")).distinct()

  gameListDF.show(10)

  val byGamePayerID = Window.partitionBy("GamePlayerID").orderBy(col("GamePlayerID"),col("PlayerGame"))
  val newgameListDF = gameListDF.withColumn("Rest_days", col("PlayerDay") - lag("PlayerDay",1,1).over(byGamePayerID))

  val numOfPlayerGamePlayDF = newplayListDF
    .filter(col("play_PlayerKey") === "26624")
    .groupBy(col("play_PlayerKey"),col("play_GameID"),col("PlayerGame"))
    .agg(
      countDistinct(col("RosterPosition")).as("NumRosterPosition"),
      countDistinct(col("PlayType")).as("NumPlayTypePerGame"),
      countDistinct(col("Position")).as("NumPositionPerGame"),
      countDistinct(col("PlayerGamePlay")).as("MaxPlayGamePlay")
    )
    .join(
      newgameListDF,
      col("play_GameID") === newgameListDF.col("GameID")
    )

  numOfPlayerGamePlayDF.show(10)//printSchema()

 */

   val parquetFileDF = schemaObj.readParquetFile("src/main/resources/data/playerGameAggInputColDF.parquet")
  parquetFileDF.printSchema()

  val parquetFileDF2 = schemaObj.readParquetFile("src/main/resources/data/playerAggInputDF.parquet")
  parquetFileDF2.printSchema()
  //parquetFileDF.show(2)

  val aggDF = parquetFileDF.groupBy("play_PlayerKey")
    .agg(
      avg("Temperature").as("avg_Temperature")
    ).withColumnRenamed("play_PlayerKey","play_PlayerKey_temp")
  aggDF.printSchema()

  val combinedDF = parquetFileDF2.join(aggDF,col("play_PlayerKey") === col("play_PlayerKey_temp"))

  combinedDF.select("play_PlayerKey","avg_NumRosterPosition","avg_NumPlayTypePerGame","avg_NumPositionPerGame",
    "avg_Temperature","avg_MaxPlayGamePlay","avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity","sum_Injury",
    "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open","outdoor","Natural","Synthetic"
  ).write.mode("append").parquet("src/main/resources/data/playerAggInputDF2.parquet")



 /*
  combinedDF.select("play_PlayerKey","avg_NumRosterPosition","avg_NumPlayTypePerGame","avg_NumPositionPerGame",
    "avg_MaxPlayGamePlay","avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity","IsInjury",
    "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open","outdoor","Natural","Synthetic"
  ).write.mode("append").parquet("src/main/resources/data/playerAggInputDF.parquet")
*/
  // Parquet files can also be used to create a temporary view and then used in SQL statements

  //println(s"The injuryDF has ${parquetFileDF.count()} rows")
  //val namesDF = spark.sql("SELECT name FROM parquetFile WHERE age BETWEEN 13 AND 19")
  //namesDF.map(attributes => "Name: " + attributes(0)).show()
}
