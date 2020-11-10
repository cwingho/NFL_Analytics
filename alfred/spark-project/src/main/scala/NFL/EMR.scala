package NFL

import org.apache.log4j.{Level, Logger}
import NFLPlayerAggDataModel.createNFLPlayerAggDataModel
import NFLPlayerGameAggDataModel.createNFLPlayerGameAggDataModel
import org.apache.spark.sql.SparkSession

object EMR {
  def main(args: Array[String]): Unit ={

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    //var filePath1 = "src/main/resources/data/playerAggInputDF2.parquet"
    //var filePath2 = "src/main/resources/data/playerGameAggInputColDF.parquet"
    val filePath1 = "s3://nfl45745/input/playerAggInputDF2.parquet"
    val filePath2 = "s3://nfl45745/input/playerGameAggInputColDF.parquet"

    val spark = SparkSession.builder()
      .appName("Data Sources and Formats")
      //.config("spark.master", "local") //Comment if run in AWS EMR
      .getOrCreate()

    //val playerAggInputDF = spark.read.parquet(filePath1)
    //println("Total records: " + playerAggInputDF.count())

    val NFLPayerAggModelObj = new createNFLPlayerAggDataModel(spark)
    NFLPayerAggModelObj.run(filePath1)

    val NFLPayerGameAggModelObj = new createNFLPlayerGameAggDataModel(spark)
    NFLPayerGameAggModelObj.run(filePath2)
  }
}