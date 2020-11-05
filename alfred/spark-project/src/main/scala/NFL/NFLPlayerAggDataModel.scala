package NFL

import org.apache.log4j.{Level, Logger}
import NFLSchema.dataSchema
import NFLModel.ModelProcessing
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
object NFLPlayerAggDataModel extends App{

  Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
  var filePath = "src/main/resources/data/playerAggInputDF2.parquet"

  var schemaObj = new dataSchema()

  var playerAggInputDF = schemaObj.readParquetFile(filePath)
  //playerAggInputDF.printSchema()

  //playerAggInputDF.groupBy("sum_Injury").count().show()
  val playerAggInputDF2 = playerAggInputDF.withColumn("IsInjury", when(col("sum_Injury") >0,1).otherwise(0))
  //playerAggInputDF2.groupBy("IsInjury").count().show()

  /*
  var injuryFilePath = "src/main/resources/data/injuryRecord.csv"
  var injuryDF = schemaObj.readFile(injuryFilePath, schemaObj.injurySchema)
  val newInjuryDF = schemaObj.constructInjuryDF(injuryDF)
  newInjuryDF.groupBy("Severity").count().show()

  var playListFilePath = "src/main/resources/data/PlayList.csv"
  val playListDF = schemaObj.readFile(playListFilePath,schemaObj.playListSchema).select("play_PlayerKey").distinct()
  var cnt = playListDF.count()
  println(cnt)
  */

  val modelInputColDF = playerAggInputDF2.select("avg_NumRosterPosition",
    "avg_NumPlayTypePerGame","avg_NumPositionPerGame","avg_MaxPlayGamePlay","avg_Temperature",
    "avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity",
    "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open",
    "outdoor","Natural","Synthetic","IsInjury").na.fill(0)

  //modelInputColDF.printSchema()

  val nflModel = new ModelProcessing()
  //Logistic Regression Model

  val lr_featuresAssembler = nflModel.createAssembler(Array("avg_NumRosterPosition",
    "avg_NumPlayTypePerGame","avg_NumPositionPerGame","avg_MaxPlayGamePlay","avg_Temperature",
    "avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity",
    "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open",
    "outdoor","Natural","Synthetic"))

  val lr_pipelineStages = Array(lr_featuresAssembler,nflModel.createLogisticRegressionModel
  ("IsInjury","features",100,0.6))

  val lr_pipeline_model = new Pipeline().setStages(lr_pipelineStages)

  nflModel.pipelineBuild(lr_pipeline_model,modelInputColDF, "LogisticRegression")

  //RandomForest Model
  val rf_featuresAssembler = nflModel.createAssembler(Array("avg_NumRosterPosition",
    "avg_NumPlayTypePerGame","avg_NumPositionPerGame","avg_MaxPlayGamePlay","avg_Temperature",
    "avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity",
    "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open",
    "outdoor","Natural","Synthetic"))

  val rf_pipelineStages = Array(rf_featuresAssembler,nflModel.createRandomForestModel
  ("IsInjury","features",100,10))


  val rf_pipeline_model = new Pipeline().setStages(rf_pipelineStages)
  nflModel.pipelineBuild(rf_pipeline_model,modelInputColDF, "RandomForest")


}
