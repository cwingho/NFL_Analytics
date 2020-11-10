package NFL

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
//https://medium.com/expedia-group-tech/deep-dive-into-apache-spark-window-functions-7b4e39ad3c86
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel,BinaryLogisticRegressionSummary}
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
    def createOneHotEncode(inputColumnsArray: Array[String]): OneHotEncoder = {
      val indexColumnArray = for (e <- inputColumnsArray) yield e + "_index"
      val outputColumnArray = for (e <- inputColumnsArray) yield e + "_vec"
      return new OneHotEncoder().setInputCols(indexColumnArray)
        .setOutputCols(outputColumnArray)
    }

    def createAssembler(inputColumnsArray: Array[String]): VectorAssembler ={
      return new VectorAssembler().setInputCols(inputColumnsArray).setOutputCol("features")
    }

    def createLogisticRegressionModel(labelColumnName:String,InputColumnName:String, maxIter:Integer, regParam: Double):LogisticRegression ={
      return new LogisticRegression()
        .setLabelCol(labelColumnName)
        .setFeaturesCol(InputColumnName)
        .setMaxIter(maxIter)
        .setRegParam(regParam)
        .setElasticNetParam(0.0)
    }

    def createRandomForestModel(labelColumnName:String,InputColumnName:String, numTrees: Integer, maxDept:Integer):RandomForestClassifier ={
      return new RandomForestClassifier()
        .setLabelCol(labelColumnName)
        .setFeaturesCol(InputColumnName)
        .setNumTrees(numTrees) // Default is 20
        .setMaxDepth(maxDept) // Default is 5
    }

    def pipelineBuild_CV(pipeLineInput:Pipeline, modelInputDF: DataFrame, paramGrid:Array[ParamMap], modelType:String):Unit ={

      val Array(trainingData, testData) = modelInputDF.randomSplit(Array(0.7, 0.3))

      val evaluator = new BinaryClassificationEvaluator()
        .setLabelCol("IsInjury")
        .setRawPredictionCol("rawPrediction")
        .setMetricName("areaUnderROC")

      val cv = new CrossValidator()
        .setEstimator(pipeLineInput)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(4)  // Use 3+ in practice
        .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

      // Run cross-validation, and choose the best set of parameters.
      val cvModel = cv.fit(trainingData)

      val predictions = cvModel.transform(testData)

      val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
      //predictions.show(5)
      if(modelType == "LogisticRegression"){

        val lrModel = bestPipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]

        println("MaxIter = " + lrModel.getMaxIter)
        println("regParam = " + lrModel.getRegParam)
        println("fitIntercept = " + lrModel.getFitIntercept)
        println("elasticNetParam = " + lrModel.getElasticNetParam)

        // Get output schema of our fitted pipeline
        val schema = bestPipelineModel.transform(modelInputDF).schema
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

        val trainingSummary = lrModel.summary

        // for multiclass, we can inspect metrics on a per-label basis
        println("False positive rate by label:")
        trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
          println(s"label $label: $rate")
        }

        println("True positive rate by label:")
        trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
          println(s"label $label: $rate")
        }

        println("Precision by label:")
        trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
          println(s"label $label: $prec")
        }

        println("Recall by label:")
        trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
          println(s"label $label: $rec")
        }


        println("F-measure by label:")
        trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
          println(s"label $label: $f")
        }

        val accuracy = trainingSummary.accuracy
        val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
        val truePositiveRate = trainingSummary.weightedTruePositiveRate
        val fMeasure = trainingSummary.weightedFMeasure
        val precision = trainingSummary.weightedPrecision
        val recall = trainingSummary.weightedRecall
        println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
          s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")
      }
      else {
        val rfModel = bestPipelineModel.stages.last.asInstanceOf[RandomForestClassificationModel]

        println("MaxDepth = " + rfModel.getMaxDepth)
        println("NumTrees = " + rfModel.getNumTrees)

        // Get output schema of our fitted pipeline
        val rf_schema = bestPipelineModel.transform(modelInputDF).schema
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

      val roc_test = evaluator.evaluate(predictions)
      println(s"ROC: $roc_test")
    }


    def pipelineBuild(pipeLineInput:Pipeline, modelInputDF: DataFrame, modelType:String):Unit ={

      val Array(trainingData, testData) = modelInputDF.randomSplit(Array(0.5, 0.5))

      val pipeline_model = pipeLineInput.fit(trainingData)

      val predictions = pipeline_model.transform(testData)

      //predictions.show(5)
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
      val evaluator = new BinaryClassificationEvaluator()
        .setLabelCol("IsInjury")
        .setRawPredictionCol("rawPrediction")
        .setMetricName("areaUnderROC")
      val roc_test = evaluator.evaluate(predictions)
      println(s"ROC: $roc_test")
    }
  }


}




