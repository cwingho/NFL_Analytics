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
import org.apache.spark.SparkConf
import NFLModel.ModelProcessing

object NFLTrackData extends App{

  Logger.getLogger("org.apache.spark").setLevel(Level.OFF)


  val spark = SparkSession.builder()
    .appName("Data Sources and Formats")
    .config("spark.master", "local")
    .getOrCreate()

  var schemaObj = new dataSchema(spark)
  val playerTrackDF = spark.read
    .schema(schemaObj.playerTrackSchema)
    .option("header", "true")
    .option("sep", ",")
    .csv("src/main/resources/data/PlayerTrackData.csv")

  //33474-14
  //34259-16
  //36621-10
  //36757-4
  //39650-16

  val tmpDF = playerTrackDF.filter(col("PlayKey") === "26624-1-1" && col("Event") =!= "").
                            withColumn("o_minus_dir",abs(col("o")-col("dir")))
  tmpDF.show()

  val byPayerKeyID = Window.partitionBy("PlayKey").orderBy(col("time"))
  val newtmpDF = tmpDF.withColumn("x_distance", col("x") - lag("x",1).over(byPayerKeyID))
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
  newtmpDF2.show()
}
