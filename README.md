# sparkmlperf
Run Spark load tests using tweaked ML/MLLib algorithms

To run :  
   mvn exec:java  -DskipTests -Dexec.mainClass="com.blazedb.spark.perf.LBFGSRunner" -Dexec.args="local[4] 100000 4"


