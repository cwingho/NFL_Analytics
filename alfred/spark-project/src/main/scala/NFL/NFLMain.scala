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

object NFLMain extends App{

  Logger.getLogger("org.apache.spark").setLevel(Level.OFF)

  var injureyFilePath = "src/main/resources/data/injuryRecord.csv"
  var playListFilePath = "src/main/resources/data/PlayList.csv"
  var playTrackFilePath = "src/main/resources/data/PlayerTrackData.csv"

  var schemaObj = new dataSchema()

  var injuryDF = schemaObj.readFile(injureyFilePath, schemaObj.injurySchema)
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

  numOfPlayerGamePlayDF.printSchema()

  var playerTrackDF = schemaObj.readFile(playTrackFilePath, schemaObj.playerTrackSchema)
  var newplayerTrackDF = schemaObj.constructPlayerTrackDF(playerTrackDF)
  //val tmpDF = playerTrackDF.filter(col("PlayKey") === "26624-1-1" && col("Event") =!= "")
 //   withColumn("o_minus_dir",abs(col("o")-col("dir")))
 // tmpDF.show()

  val tempDF = numOfPlayerGamePlayDF.join(newplayerTrackDF,
    col("play_GameID") === newplayerTrackDF.col("PlayerTrack_GameID"),"leftouter")
    .select("play_PlayerKey","play_GameID", "NumRosterPosition","NumPlayTypePerGame","NumPositionPerGame",
      "MaxPlayGamePlay","RosterPosition","StadiumType","FieldType","Temperature","Weather","Rest_days","avg_o_mins_dir","avg_x_velocity",
    "avg_y_velocity","avg_velocity","IsInjury")

  tempDF.show(5)

  tempDF.write.mode("append").parquet("src/main/resources/data/modelInputColDF.parquet")

  /*
  val modelInputColDF = tempDF.select("RosterPosition",
    "StadiumType","FieldType","Temperature","Weather","Rest_days","NumRosterPosition",
    "NumPlayTypePerGame","NumPositionPerGame","MaxPlayGamePlay","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity","IsInjury")



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


