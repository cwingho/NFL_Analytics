package NFL

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{FloatType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object NFLSchema {

  class dataSchema(sparkSession: SparkSession){

    val spark = sparkSession

    val playListSchema = StructType(Array(
      StructField("play_PlayerKey", StringType),
      StructField("play_GameID", StringType),
      StructField("play_PlayKey", StringType),
      StructField("RosterPosition", StringType),
      StructField("PlayerDay", IntegerType),
      StructField("PlayerGame", IntegerType),
      StructField("StadiumType", StringType),
      StructField("FieldType", StringType),
      StructField("Temperature", FloatType),
      StructField("Weather", StringType),
      StructField("PlayType", StringType),
      StructField("PlayerGamePlay", IntegerType),
      StructField("Position", StringType),
      StructField("PositionGroup", StringType)
    ))
    // InjuryRecord schema
    //PlayerKey,GameID,PlayKey,BodyPart,Surface,DM_M1,DM_M7,DM_M28,DM_M42
    val injurySchema = StructType(Array(
      StructField("PlayerKey", StringType),
      StructField("GameID", StringType),
      StructField("PlayKey", StringType),
      StructField("BodyPart", StringType),
      StructField("Surface", StringType),
      StructField("DM_M1", IntegerType),
      StructField("DM_M7", IntegerType),
      StructField("DM_M28", IntegerType),
      StructField("DM_M42", IntegerType)
    ))
    //PlayerTrackData schema
    //PlayKey,time,event,x,y,dir,dis,o,s
    val playerTrackSchema = StructType(Array(
      StructField("PlayKey", StringType),
      StructField("time", FloatType),
      StructField("event", StringType),
      StructField("x", FloatType),
      StructField("y", FloatType),
      StructField("dir", FloatType),
      StructField("dis", FloatType),
      StructField("o", FloatType),
      StructField("s",FloatType)
    ))

    //StadiumType
    val dome_open = List("Domed, Open","Domed, open")
    val dome_closed = List("Dome","Domed, closed","Closed Dome","Domed","Dome, closed")
    val indoor_open = List("Indoor, Open Roof", "Open", "Retr. Roof-Open", "Retr. Roof - Open")
    val indoor_closed = List("Indoors", "Indoor", "Indoor, Roof Closed", "Indoor, Roof Closed","Retractable Roof","Retr. Roof-Closed",
      "Retr. Roof - Closed", "Retr. Roof Closed")
    val outdoor = List("Outdoor", "Outdoors", "Cloudy", "Heinz Field", "Outdor","Ourdoor","Outside","Outddors", "Outdoor Retr Roof-Open","Oudoor","Bowl")

    //Weather
    val overcast = List("Party Cloudy", "Cloudy, chance of rain","Coudy","Cloudy and cold",
      "Cloudy, fog started developing in 2nd quarter","Partly Clouidy", "Mostly Coudy", "Cloudy and Cool",
      "cloudy", "Partly cloudy", "Overcast", "Hazy", "Mostly cloudy", "Mostly Cloudy",
      "Partly Cloudy", "Cloudy")
    val clear = List("Partly clear", "Sunny and clear", "Sun & clouds", "Clear and Sunny","Sunny, Windy",
      "Sunny and cold", "Sunny Skies", "Clear and Cool", "Clear and sunny",
      "Sunny, highs to upper 80s", "Mostly Sunny Skies", "Cold","Clear and Cool",
      "Clear and warm", "Sunny and warm", "Clear and cold", "Mostly sunny",
      "T: 51; H: 55; W: NW 10 mph", "Clear Skies", "Clear skies", "Partly sunny",
      "Fair", "Partly Sunny", "Mostly Sunny", "Clear", "Sunny","Clear to Partly Cloudy")
    val rain = List("30% Chance of Rain", "Rainy", "Rain Chance 40%", "Showers", "Cloudy, 50% change of rain",
      "Rain likely, temps in low 40s.","Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.",
      "Scattered Showers","Cloudy, Rain", "Rain shower", "Light Rain", "Rain","10% Chance of Rain")
    val snow = List("Cloudy, light snow accumulating 1-3\"\"", "Heavy lake effect snow", "Snow")
    val weathermissing = List("N/A Indoor", "Indoors", "Indoor", "N/A (Indoors)", "Controlled Climate","Heat Index 95")

    var stringColListInPlayer = List("PlayType","FieldType","Weather","Position","PositionGroup","StadiumType")

    def readFile(filePath:String, schema:StructType): DataFrame ={
      return spark.read
        .schema(schema)
        .option("header", "true")
        .option("sep", ",")
        .csv(filePath)
    }

    def readParquetFile(filePath:String): DataFrame ={
      return spark.read.parquet(filePath)
    }

    def constructInjuryDF(injuryDF:DataFrame):DataFrame ={
      val newInjuryDF = injuryDF
        .withColumnRenamed("PlayerKey", "InjPlayerKey")
        .withColumnRenamed("GameID", "InjGameID")
        .withColumnRenamed("PlayKey", "InjPlayKey")
        .withColumn("Severity", col("DM_M1") + col("DM_M7") +
                                              col("DM_M28") + col("DM_M42"))
      return newInjuryDF
    }

    def constructPlayList(playListDF:DataFrame):DataFrame = {

      val newplayListDF = playListDF.na.fill("Missing", stringColListInPlayer)
        .withColumn("PlayType", when(col("PlayType") ==="0","Missing").otherwise(col("PlayType")))
        .withColumn("StadiumType", when(col("StadiumType").isin(dome_open:_*),"dome_open")
          .when(col("StadiumType").isin(dome_closed:_*),"dome_closed")
          .when(col("StadiumType").isin(indoor_open:_*),"indoor_open")
          .when(col("StadiumType").isin(indoor_closed:_*),"indoor_closed")
          .when(col("StadiumType").isin(outdoor:_*),"outdoor")
          .otherwise(col("StadiumType")))
        .withColumn("Weather", when(col("Weather").isin(overcast:_*),"overcast")
          .when(col("Weather").isin(clear:_*),"clear")
          .when(col("Weather").isin(rain:_*),"rain")
          .when(col("Weather").isin(snow:_*),"snow")
          .when(col("Weather").isin(weathermissing:_*),"Missing")
          .otherwise(col("Weather")))
        .withColumn("Temperature",
          when(col("Temperature") < 0, 61).otherwise(col("Temperature")))

      val gameListDF = newplayListDF.select(col("play_PlayerKey").as("GamePlayerID"),
        col("play_GameID").as("GameID"), col("PlayerDay"),
        col("PlayerGame"), col("RosterPosition"),col("StadiumType"),
        col("FieldType"),col("Temperature"),col("Weather")).distinct()

      val byGamePayerID = Window.partitionBy("GamePlayerID").orderBy(col("GamePlayerID"),col("PlayerGame"))
      val newgameListDF = gameListDF.withColumn("Rest_days", col("PlayerDay") - lag("PlayerDay",1,1)
                                    .over(byGamePayerID))

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
      return numOfPlayerGamePlayDF
    }

    def constructPlayerTrackDF(playerTrackDF:DataFrame):DataFrame = {

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
          concat(col("PlayKeyTmp").getItem(0),lit("-"), col("PlayKeyTmp").getItem(1)).as("PlayerTrack_GameID"),
          col("PlayKey"), col("o_minus_dir"),
          col("x_velocity"),col("y_velocity"),
          col("velocity")
        ).groupBy(col("PlayerTrack_GameID"))
        .agg(
          avg(col("o_minus_dir")).as("avg_o_mins_dir"),
          avg(col("x_velocity")).as("avg_x_velocity"),
          avg(col("y_velocity")).as("avg_y_velocity"),
          avg(col("velocity")).as("avg_velocity")
        )
      return newtmpDF2
    }




  }
}
