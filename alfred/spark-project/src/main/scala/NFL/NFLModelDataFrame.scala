package NFL

import org.apache.spark.sql.functions.{col, column, expr}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import NFLSchema.dataSchema
import org.apache.spark.sql.SparkSession


object NFLModelDataFrame extends App{

  var injuryFilePath = "src/main/resources/data/injuryRecord.csv"
  var playListFilePath = "src/main/resources/data/PlayList.csv"
  var playTrackFilePath = "src/main/resources/data/PlayerTrackData.csv"


  val spark = SparkSession.builder()
    .appName("Data Sources and Formats")
    //.config("spark.master", "local") //Comment if run in AWS EMR
    .getOrCreate()

  var schemaObj = new dataSchema(spark)

  var injuryDF = schemaObj.readFile(injuryFilePath, schemaObj.injurySchema)
  val newInjuryDF = schemaObj.constructInjuryDF(injuryDF)

  val playListDF = schemaObj.readFile(playListFilePath,schemaObj.playListSchema)
  var newplayListDF = schemaObj.constructPlayList(playListDF)

  val numOfPlayerGamePlayDF = newplayListDF
    .join(
      newInjuryDF.select("InjPlayerKey","InjGameID","Severity"),
      col("GamePlayerID") === col("InjPlayerKey") && col("GameID") === col("InjGameID"),
      "leftouter"
    ).withColumn("IsInjury", when(col("Severity") > 0, 1).otherwise(0))
    .orderBy(col("play_PlayerKey"),col("play_GameID"))

  //numOfPlayerGamePlayDF.printSchema()

  var playerTrackDF = schemaObj.readFile(playTrackFilePath, schemaObj.playerTrackSchema)
  var newplayerTrackDF = schemaObj.constructPlayerTrackDF(playerTrackDF)

  val tempDF = numOfPlayerGamePlayDF.join(newplayerTrackDF,
    col("play_GameID") === newplayerTrackDF.col("PlayerTrack_GameID"),"leftouter")
    .select("play_PlayerKey","play_GameID", "NumRosterPosition","NumPlayTypePerGame","NumPositionPerGame",
      "MaxPlayGamePlay","RosterPosition","StadiumType","FieldType","Temperature","Weather","Rest_days","avg_o_mins_dir","avg_x_velocity",
      "avg_y_velocity","avg_velocity","IsInjury")


  tempDF.write.mode("append").parquet("src/main/resources/data/playerGameAggInputColDF.parquet")

  val aggDF = tempDF.groupBy("play_PlayerKey")
    .agg(
      avg("NumRosterPosition").as("avg_NumRosterPosition"),
      avg("NumPlayTypePerGame").as("avg_NumPlayTypePerGame"),
      avg("NumPositionPerGame").as("avg_NumPositionPerGame"),
      avg("MaxPlayGamePlay").as("avg_MaxPlayGamePlay"),
      avg("Rest_days").as("avg_Rest_days"),
      avg("avg_o_mins_dir").as("avg_o_mins_dir"),
      avg("avg_x_velocity").as("avg_x_velocity"),
      avg("avg_y_velocity").as("avg_y_velocity"),
      avg("avg_velocity").as("avg_velocity"),
      avg("Temperature").as("avg_Temperature"),
      sum("IsInjury").as("sum_Injury")
    )

  val pivot_weatherDF = tempDF
    .select("play_PlayerKey","play_GameID","Weather")
    .groupBy("play_PlayerKey").pivot("Weather").count()
    .withColumnRenamed("play_PlayerKey","play_PlayerKey_w")

  val pivot_stadiumTypeDF = tempDF
    .select("play_PlayerKey","play_GameID","StadiumType")
    .groupBy("play_PlayerKey").pivot("StadiumType").count()
    .withColumnRenamed("play_PlayerKey","play_PlayerKey_s")

  val pivot_fieldTypeDF = tempDF
    .select("play_PlayerKey","play_GameID","FieldType")
    .groupBy("play_PlayerKey").pivot("FieldType").count()
    .withColumnRenamed("play_PlayerKey","play_PlayerKey_f")


  val combinedDF = aggDF
    .join(
      pivot_weatherDF,
      col("play_PlayerKey") === col("play_PlayerKey_w"),
      "leftouter"
    )
    .join(
      pivot_stadiumTypeDF,
      col("play_PlayerKey") === col("play_PlayerKey_s"),
      "leftouter"
    )
    .join(
      pivot_fieldTypeDF,
      col("play_PlayerKey") === col("play_PlayerKey_f"),
      "leftouter"
    ).drop("play_PlayerKey_w","play_PlayerKey_s","play_PlayerKey_f")

  combinedDF.printSchema()
  combinedDF.show(5)
  combinedDF.select("play_PlayerKey","avg_NumRosterPosition","avg_NumPlayTypePerGame","avg_NumPositionPerGame",
    "avg_Temperature","avg_MaxPlayGamePlay","avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity","sum_Injury",
    "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open","outdoor","Natural","Synthetic"
  ).write.mode("append").parquet("src/main/resources/data/playerAggInputDF2.parquet")


}



