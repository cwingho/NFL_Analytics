
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, column, expr}
import org.apache.spark.sql.types._

object NFLDataFrames extends App {
  val spark = SparkSession.builder()
    .appName("Data Sources and Formats")
    .config("spark.master", "local")
    .getOrCreate()

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

  val injuryDF = spark.read
    .schema(injurySchema)
    .option("header", "true")
    .option("sep", ",")
    .csv("src/main/resources/data/injuryRecord.csv")

  injuryDF.printSchema()
  println(s"The injuryDF has ${injuryDF.count()} rows")

  // adding a column
  val newInjuryDF = injuryDF
    .withColumnRenamed("PlayerKey", "InjPlayerKey")
    .withColumnRenamed("GameID", "InjGameID")
    .withColumnRenamed("PlayKey", "InjPlayKey")
    .withColumn("Severity", col("DM_M1") + col("DM_M7") + col("DM_M28") + col("DM_M42"))
  newInjuryDF.show(5)

  //PlayList schema
  //PlayerKey,GameID,PlayKey,RosterPosition,PlayerDay,PlayerGame,StadiumType,
  //FieldType,Temperature,Weather,PlayType,PlayerGamePlay,Position,PositionGroup

  val playListSchema = StructType(Array(
    StructField("PlayerKey", StringType),
    StructField("GameID", StringType),
    StructField("PlayKey", StringType),
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
    StructField("PositionGroup", StringType),
  ))

  val playListDF = spark.read
    .schema(playListSchema)
    .option("header", "true")
    .option("sep", ",")
    .csv("src/main/resources/data/PlayList.csv")

  playListDF.printSchema()
  println(s"The PlayListDF has ${playListDF.count()} rows")


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

  val playerTrackDF = spark.read
    .schema(playerTrackSchema)
    .option("header", "true")
    .option("sep", ",")
    .csv("src/main/resources/data/PlayerTrackData.csv")

  playerTrackDF.printSchema()
  println(s"The PlayerTrackDF has ${playerTrackDF.count()} rows")

  //playerTrackDF.show(10)

  //Different body part and severity injury
  val countByBodyPartDF = newInjuryDF
    .groupBy(col("BodyPart"), col("Severity")) // includes null
    .count()
    .orderBy("BodyPart", "Severity")
  countByBodyPartDF.show()
  //The injury count by different surface
  val countBySurfaceDF = newInjuryDF
    .groupBy(col("Surface"), col("Severity")) // includes null
    .count()
    .orderBy("Surface", "Severity")
  countBySurfaceDF.show()

  //injuryPlayListDF: The game history of each injury player
  val injuryAllPlayListDF = newInjuryDF.join(playListDF,
    newInjuryDF.col("InjPlayerKey") === playListDF.col("PlayerKey"),
    "inner")

  injuryAllPlayListDF.printSchema()
  //injuryAllPlayListDF.show(10)

  val injuryAllGameHistoryPlayListDF = injuryAllPlayListDF.select(
    "InjPlayerKey", "PlayKey",
    "RosterPosition", "PlayerDay", "PlayerGame",
    "StadiumType","FieldType","Temperature","Weather",
    "PlayType","PlayerGamePlay","Position","PositionGroup")
    .orderBy("InjPlayerKey","PlayKey","PlayerGamePlay")

  println(s"The Injury PlayList History has ${injuryAllGameHistoryPlayListDF.count()} rows")

  val injuryGamePlayListDF = injuryAllPlayListDF.select(
    "InjPlayerKey", "InjGameID", "PlayKey",
    "RosterPosition", "PlayerDay", "PlayerGame",
    "StadiumType","FieldType","Temperature","Weather",
    "PlayType","PlayerGamePlay","Position","PositionGroup")
    .where(col("InjGameID") === col("GameID"))
    .orderBy("InjPlayerKey","InjGameID","PlayerGamePlay")

  println(s"The Injury PlayList has ${injuryGamePlayListDF.count()} rows")
  //injuryGamePlayListDF.show()
}
