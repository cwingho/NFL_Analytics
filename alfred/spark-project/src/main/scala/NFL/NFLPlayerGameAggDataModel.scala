package NFL

import NFLSchema.dataSchema
import NFLModel.ModelProcessing
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
object NFLPlayerGameAggDataModel extends App{

  class createNFLPlayerGameAggDataModel(sparkSession: SparkSession)
  {
    var schemaObj = new dataSchema(sparkSession)

    def run(filePath: String): Unit =
    {
      var playerGameAggInputColDF = schemaObj.readParquetFile(filePath)

      val modelInputColDF = playerGameAggInputColDF.select("RosterPosition",
        "StadiumType","FieldType","Temperature","Weather","Rest_days","NumRosterPosition",
        "NumPlayTypePerGame","NumPositionPerGame","MaxPlayGamePlay",
        "avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity","IsInjury")

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
        weatherIndexer, oneHotEncoder, lr_featuresAssembler,nflModel.createLogisticRegressionModel
        ("IsInjury","features",100, 0.5))

      val lr_pipeline_model = new Pipeline().setStages(lr_pipelineStages)


      nflModel.pipelineBuild(lr_pipeline_model,modelInputColDF, "LogisticRegression")

      //RandomForest Model
      val rf_featuresAssembler = nflModel.createAssembler(Array("RosterPosition_index", "StadiumType_index", "FieldType_index", "Temperature", "Weather_index",
        "Rest_days","NumRosterPosition","NumPlayTypePerGame","NumPositionPerGame", "MaxPlayGamePlay","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity"))

      val rf_pipelineStages = Array(rosterPositionIndexer, stadiumTypeIndexer, fieldTypeIndexer,
        weatherIndexer, rf_featuresAssembler,nflModel.createRandomForestModel
        ("IsInjury","features",200, 20))

      val rf_pipeline_model = new Pipeline().setStages(rf_pipelineStages)

      nflModel.pipelineBuild(rf_pipeline_model,modelInputColDF, "RandomForest")

    }
  }
}
