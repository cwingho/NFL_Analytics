package NFL

import org.apache.log4j.{Level, Logger}
import NFLSchema.dataSchema
import NFLModel.ModelProcessing
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

object NFLPlayerAggDataModel extends App{

  class createNFLPlayerAggDataModel(sparkSession: SparkSession)
  {
    var schemaObj = new dataSchema(sparkSession)

    def run(filePath:String): Unit =
    {
      var playerAggInputDF = schemaObj.readParquetFile(filePath)

      val playerAggInputDF2 = playerAggInputDF.withColumn("IsInjury", when(col("sum_Injury") >0,1).otherwise(0))

      val modelInputColDF = playerAggInputDF2.select("avg_NumRosterPosition",
        "avg_NumPlayTypePerGame","avg_NumPositionPerGame","avg_MaxPlayGamePlay","avg_Temperature",
        "avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity",
        "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open",
        "outdoor","Natural","Synthetic","IsInjury").na.fill(0)

      val nflModel = new ModelProcessing()
      //Logistic Regression Model

      val lr_featuresAssembler = nflModel.createAssembler(Array("avg_NumRosterPosition",
        "avg_NumPlayTypePerGame","avg_NumPositionPerGame","avg_MaxPlayGamePlay","avg_Temperature",
        "avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity",
        "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open",
        "outdoor","Natural","Synthetic"))

      val lrModel = nflModel.createLogisticRegressionModel("IsInjury","features",
        100,0.6)
      val lr_pipelineStages = Array(lr_featuresAssembler,lrModel)

      val lr_pipeline_model = new Pipeline().setStages(lr_pipelineStages)

      val lr_paramGrid = new ParamGridBuilder()
        .addGrid(lrModel.regParam, Array(0.6, 0.62))
        .addGrid(lrModel.elasticNetParam, Array(0.0, 0.1))
        .addGrid(lrModel.fitIntercept, Array(true, false))
        .addGrid(lrModel.maxIter, Array(58, 60, 62))
        .build()

      nflModel.pipelineBuild_CV(lr_pipeline_model, modelInputColDF, lr_paramGrid, "LogisticRegression")

      //RandomForest Model

      val rf_featuresAssembler = nflModel.createAssembler(Array("avg_NumRosterPosition",
        "avg_NumPlayTypePerGame","avg_NumPositionPerGame","avg_MaxPlayGamePlay","avg_Temperature",
        "avg_Rest_days","avg_o_mins_dir","avg_x_velocity","avg_y_velocity","avg_velocity",
        "clear","overcast","rain","snow","dome_closed","dome_open","indoor_closed","indoor_open",
        "outdoor","Natural","Synthetic"))

      val rfModel = nflModel.createRandomForestModel("IsInjury","features",
        100,10)
      val rf_pipelineStages = Array(rf_featuresAssembler, rfModel)

      val rf_paramGrid = new ParamGridBuilder()
        .addGrid(rfModel.maxDepth, Array(10, 15,20))
        .addGrid(rfModel.numTrees, Array(50, 60, 70))
        .build()
      val rf_pipeline_model = new Pipeline().setStages(rf_pipelineStages)
      nflModel.pipelineBuild_CV(rf_pipeline_model,modelInputColDF, rf_paramGrid, "RandomForest")

    }

  }




}
