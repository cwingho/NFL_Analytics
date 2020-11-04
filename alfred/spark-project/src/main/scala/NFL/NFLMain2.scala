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

object NFLMain2 extends App{

  Logger.getLogger("org.apache.spark").setLevel(Level.OFF)

  var injureyFilePath = "src/main/resources/data/injuryRecord.csv"
  var playListFilePath = "src/main/resources/data/PlayList.csv"
  var playTrackFilePath = "src/main/resources/data/PlayerTrackData.csv"

  var schemaObj = new dataSchema()
  var injuryDF = schemaObj.readFile(injureyFilePath, schemaObj.injurySchema)
  injuryDF.printSchema()

  // adding a column
  val newInjuryDF = injuryDF
    .withColumnRenamed("PlayerKey", "InjPlayerKey")
    .withColumnRenamed("GameID", "InjGameID")
    .withColumnRenamed("PlayKey", "InjPlayKey")
    .withColumn("Severity", col("DM_M1") + col("DM_M7") +
      col("DM_M28") + col("DM_M42"))
  newInjuryDF.show(5)

  val playListDF = schemaObj.readFile(playListFilePath,schemaObj.playListSchema)

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
    .withColumn("Temperature",
      when(col("Temperature") < 0, 61).otherwise(col("Temperature")))

  newplayListDF.printSchema()
  //newplayListDF.groupBy(col("Weather")).count().show()
  var gameListDF = newplayListDF.select(col("play_PlayerKey").as("GamePlayerID"), col("play_GameID").as("GameID"), col("PlayerDay"),
    col("PlayerGame"), col("RosterPosition"),col("StadiumType"),
    col("FieldType"),col("Temperature"),col("Weather")).distinct()

  val byGamePayerID = Window.partitionBy("GamePlayerID").orderBy(col("GamePlayerID"),col("PlayerGame"))
  val newgameListDF = gameListDF.withColumn("Rest_days", col("PlayerDay") - lag("PlayerDay",1,1).over(byGamePayerID))

  val numOfPlayerGamePlayDF = newplayListDF
    //.filter(col("play_GameID") === "26624-1")
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
    .join(
      newInjuryDF.select("InjPlayerKey","InjGameID","Severity"),
      col("GamePlayerID") === col("InjPlayerKey") && col("GameID") === col("InjGameID"),
      "leftouter"
    ).withColumn("IsInjury", when(col("Severity") > 0, 1).otherwise(0))
    .orderBy(col("play_PlayerKey"),col("play_GameID"))

  numOfPlayerGamePlayDF.printSchema()

  var playerTrackDF = schemaObj.readFile(playTrackFilePath, schemaObj.playerTrackSchema)

  //val tmpDF = playerTrackDF.filter(col("PlayKey") === "26624-1-1" && col("Event") =!= "")
  //   withColumn("o_minus_dir",abs(col("o")-col("dir")))
  // tmpDF.show()

  val byPayerKeyID = Window.partitionBy("PlayKey").orderBy(col("time"))

  val newtmpDF = playerTrackDF.filter(col("Event") =!= "")
    //.filter(col("PlayKey") === "26624-1-1" && col("Event") =!= "")
    .withColumn("x_distance", col("x") - lag("x",1).over(byPayerKeyID))
    .withColumn("o_minus_dir",abs(col("o")-col("dir")))
    .withColumn("y_distance", col("y") - lag("y",1).over(byPayerKeyID))
    .withColumn("time_period", col("time") - lag("time",1).over(byPayerKeyID))
    .withColumn("x_velocity", col("x_distance")/col("time"))
    .withColumn("y_velocity", col("y_distance")/col("time"))
    .withColumn("velocity", sqrt(pow(col("x_velocity"),2) + pow(col("y_velocity"),2)))

  //newtmpDF.show()
  val newtmpDF2 = newtmpDF.withColumn("PlayKeyTmp", split(col("PlayKey"),"-",3))
    .select(
      //col("PlayKeyTmp").getItem(0).as("PlayID"),
      //col("PlayKeyTmp").getItem(1).as("ID"),
      concat(col("PlayKeyTmp").getItem(0),lit("-"), col("PlayKeyTmp").getItem(1)).as("GameID"),
      col("PlayKey"), col("o_minus_dir"),
      col("x_velocity"),col("y_velocity"),
      col("velocity")
    ).groupBy(col("GameID"))
    .agg(
      avg(col("o_minus_dir")).as("avg_o_mins_dir"),
      avg(col("x_velocity")).as("avg_x_velocity"),
      avg(col("y_velocity")).as("avg_y_velocity"),
      avg(col("velocity")).as("avg_velocity")
    )
  //newtmpDF2.show()

  val newDF = numOfPlayerGamePlayDF.join(newtmpDF2,
    col("play_GameID") === newtmpDF2.col("GameID"),"leftouter")

  newDF.printSchema()
  //newDF.filter()
  val pivot_weatherDF = newDF.filter(col("play_PlayerKey") === "33474")
    .select("play_PlayerKey","play_GameID","Weather")
    .groupBy("play_PlayerKey").pivot("Weather").count()

  val pivot_stadiumTypeDF = newDF.filter(col("play_PlayerKey") === "33474")
    .select("play_PlayerKey","play_GameID","StadiumType")
    .groupBy("play_PlayerKey").pivot("StadiumType").count()

  val pivot_fieldTypeDF = newDF.filter(col("play_PlayerKey") === "33474")
    .select("play_PlayerKey","play_GameID","FieldType")
    .groupBy("play_PlayerKey").pivot("FieldType").count()

  pivot_weatherDF.show()
  pivot_stadiumTypeDF.show()
  pivot_fieldTypeDF.show()
  //numOfPlayerGamePlayDF.show(5)

  /*
   val modelInputColDF = newDF.select("RosterPosition",
     "StadiumType","FieldType","Temperature","Weather","Rest_days","NumRosterPosition",
     "NumPlayTypePerGame","NumPositionPerGame","MaxPlayGamePlay","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity","IsInjury")



   modelInputColDF.show(5)

   val nflModel = new ModelProcessing()
   val rosterPositionIndexer = nflModel.createIndexer("RosterPosition")
   val stadiumTypeIndexer = nflModel.createIndexer("StadiumType")
   val fieldTypeIndexer = nflModel.createIndexer("FieldType")
   val weatherIndexer = nflModel.createIndexer("Weather")
   val oneHotEncoder = nflModel.createOneHotEncode(Array("RosterPosition","StadiumType",
     "FieldType","Weather"))

   //Logistic Regression Model
   val lr_featuresAssembler = nflModel.createAssembler(Array("RosterPosition_vec", "StadiumType_vec", "FieldType_vec", "Temperature", "Weather_vec",
     "Rest_days","NumRosterPosition","NumPlayTypePerGame","NumPositionPerGame", "MaxPlayGamePlay","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity"))

   val lr_pipelineStages = Array(rosterPositionIndexer, stadiumTypeIndexer, fieldTypeIndexer,
     weatherIndexer, oneHotEncoder, lr_featuresAssembler,nflModel.createLogisticRegressionModel("IsInjury","features"))

   val lr_pipeline_model = new Pipeline().setStages(lr_pipelineStages)

   nflModel.pipelineBuild(lr_pipeline_model,modelInputColDF, "LogisticRegression")


   //RandomForest Model
   val rf_featuresAssembler = nflModel.createAssembler(Array("RosterPosition_index", "StadiumType_index", "FieldType_index", "Temperature", "Weather_index",
     "Rest_days","NumRosterPosition","NumPlayTypePerGame","NumPositionPerGame", "MaxPlayGamePlay","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity"))

   val rf_pipelineStages = Array(rosterPositionIndexer, stadiumTypeIndexer, fieldTypeIndexer,
     weatherIndexer, rf_featuresAssembler,nflModel.createRandomForestModel("IsInjury","features"))

   val rf_pipeline_model = new Pipeline().setStages(rf_pipelineStages)

   nflModel.pipelineBuild(rf_pipeline_model,modelInputColDF, "RandomForest")

    */

}


