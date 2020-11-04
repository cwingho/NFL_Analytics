package NFL

import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
//https://medium.com/expedia-group-tech/deep-dive-into-apache-spark-window-functions-7b4e39ad3c86
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.attribute._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

object NFLModel extends App{

  class ModelProcessing() {

    //Mlib Linear Regression model
    def createIndexer(inputColumn: String): StringIndexer = {
      return new StringIndexer().setInputCol(inputColumn).setOutputCol(inputColumn + "_index")
    }

    //("RosterPosition_index","StadiumType_index","FieldType_index","Weather_index")
    //Array("RosterPosition_vec","StadiumType_vec","FieldType_vec","Weather_vec")
    def createOneHotEncode(inputColumnsArray: Array[String]): OneHotEncoderEstimator = {
      val indexColumnArray = for (e <- inputColumnsArray) yield e + "_index"
      val outputColumnArray = for (e <- inputColumnsArray) yield e + "_vec"
      return new OneHotEncoderEstimator()
        .setInputCols(indexColumnArray)
        .setOutputCols(outputColumnArray)
    }

    def createAssembler(inputColumnsArray: Array[String]): VectorAssembler ={
      return new VectorAssembler().setInputCols(inputColumnsArray).setOutputCol("features")
    }

    def createLogisticRegressionModel(labelColumnName:String,InputColumnName:String):LogisticRegression ={
      return new LogisticRegression()
        .setLabelCol(labelColumnName)
        .setFeaturesCol(InputColumnName)
        .setMaxIter(1)
        .setRegParam(0.05)
        .setElasticNetParam(0.0)
    }

    def createRandomForestModel(labelColumnName:String,InputColumnName:String):RandomForestClassifier ={
      return new RandomForestClassifier()
        .setLabelCol(labelColumnName)
        .setFeaturesCol(InputColumnName)
        .setNumTrees(2) // Default is 20
        .setMaxDepth(1) // Default is 5
    }

    def pipelineBuild(pipeLineInput:Pipeline, modelInputDF: DataFrame, modelType:String):Unit ={

      val pipeline_model = pipeLineInput.fit(modelInputDF)

      if(modelType == "LogisticRegression"){
        val lrModel = pipeline_model.stages.last.asInstanceOf[LogisticRegressionModel]

        // Get output schema of our fitted pipeline
        val schema = pipeline_model.transform(modelInputDF).schema
        // Extract the attributes of the input (features) column to our logistic regression model
        val lrfeatureAttrs = AttributeGroup.fromStructField(schema(lrModel.getFeaturesCol)).attributes.get
        val lrfeatures = lrfeatureAttrs.map(_.name.get)

        // Add "(Intercept)" to list of feature names if the model was fit with an intercept
        val lrfeatureNames: Array[String] = if (lrModel.getFitIntercept) {
          Array("(Intercept)") ++ lrfeatures
        } else {
          lrfeatures
        }
        // Get array of coefficients
        val lrModelCoeffs = lrModel.coefficients.toArray
        val lrcoeffs = if (lrModel.getFitIntercept) {
          lrModelCoeffs ++ Array(lrModel.intercept)
        } else {
          lrModelCoeffs
        }
        // Print feature names & coefficients together
        println("Feature\tCoefficient")
        lrfeatureNames.zip(lrcoeffs).foreach { case (feature, coeff) =>
          println(s"$feature\t$coeff")
        }
      }
      else {
        val rfModel = pipeline_model.stages.last.asInstanceOf[RandomForestClassificationModel]

        // Get output schema of our fitted pipeline
        val rf_schema = pipeline_model.transform(modelInputDF).schema
        // Extract the attributes of the input (features) column to our logistic regression model
        val rf_featureAttrs = AttributeGroup.fromStructField(rf_schema(rfModel.getFeaturesCol)).attributes.get
        val rf_features = rf_featureAttrs.map(_.name.get)

        // Get array of coefficients
        val rfModelImportances = rfModel.featureImportances.toArray

        // Print feature names & Importances together
        println("Feature\tImportances")
        rf_features.zip(rfModelImportances).foreach { case (feature, importance) =>
          println(s"$feature\t$importance")
        }
      }
    }
  }

}




