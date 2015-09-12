package com.blazedb.spark.perf 

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import java.util.Date

import org.apache.spark.mllib.optimization._
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random


import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint


object  LBFGSRunner {

  def main(args: Array[String]) = {
    run(args(0), Integer.parseInt(args(1)), Integer.parseInt(args(2)))
  }
  def run(master: String, numPoints: Int, maxCores: Int) : Unit = {
    val sconf = new SparkConf()
      .set("test.num.points", "" + numPoints)
       .set("spark.cores.max", ""+maxCores)
    .set("spark.io.compression.codec", "lz4")
    val sc = new SparkContext(master, getClass.getSimpleName, sconf)
    println(s"num points=${sc.getConf.get("test.num.points")}")
    run(sc)
  }

  // Copied from GradientDescentSuite
  // Generate input of the form Y = logistic(offset + scale * X)
  def generateGDInput(
      offset: Double,
      scale: Double,
      nPoints: Int,
      seed: Int): Seq[LabeledPoint]  = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val unifRand = new Random(45)
    val rLogis = (0 until nPoints).map { i =>
      val u = unifRand.nextDouble()
      math.log(u) - math.log(1.0-u)
    }

    val y: Seq[Int] = (0 until nPoints).map { i =>
      val yVal = offset + scale * x1(i) + rLogis(i)
      if (yVal > 0) 1 else 0
    }

    (0 until nPoints).map(i => LabeledPoint(y(i), Vectors.dense(x1(i))))
  }

  def run(sc: SparkContext) = {
//    val nPoints = Integer.parseInt(Option(System.getProperty("test.num.points")).getOrElse("10000"))
    val nPoints = Integer.parseInt(Option(sc.getConf.get("test.num.points")).getOrElse("10000"))
    val sdate = new Date
    System.err.println(s"Starting test at ${sdate.toString}")
    System.err.println(s"npoints=$nPoints")
    val A = 2.0
    val B = -1.5

    val initialB = -1.0
    val initialWeights = Array(initialB)

    val gradient = new LogisticGradient()
    val numCorrections = 10
    val miniBatchFrac = 1.0

    val simpleUpdater = new SimpleUpdater()
    val squaredL2Updater = new SquaredL2Updater()

    // Add an extra variable consisting of all 1.0's for the intercept.
    val testData = generateGDInput(A, B, nPoints, 42)
    val data = testData.map { case LabeledPoint(label, features) =>
      label -> Vectors.dense(1.0 +: features.toArray)
    }

    lazy val dataRDD = sc.parallelize(data, 2).cache()

    System.err.println("LBFGS loss should be decreasing and match the result of Gradient Descent.")
    val regParam = 0

    val initialWeightsWithIntercept = Vectors.dense(1.0 +: initialWeights.toArray)
    val convergenceTol = 1e-12
    val numIterations = 10

    val (_, loss) = LBFGS.runLBFGS(
      dataRDD,
      gradient,
      simpleUpdater,
      numCorrections,
      convergenceTol,
      numIterations,
      regParam,
      initialWeightsWithIntercept)

    // Since the cost function is convex, the loss is guaranteed to be monotonically decreasing
    // with L-BFGS optimizer.
    // (SGD doesn't guarantee this, and the loss will be fluctuating in the optimization process.)
    assert((loss, loss.tail).zipped.forall(_ > _), "loss should be monotonically decreasing.")

    val stepSize = 1.0
    // Well, GD converges slower, so it requires more iterations!
    val numGDIterations = 50
    val (_, lossGD) = GradientDescent.runMiniBatchSGD(
      dataRDD,
      gradient,
      simpleUpdater,
      stepSize,
      numGDIterations,
      regParam,
      miniBatchFrac,
      initialWeightsWithIntercept)

    // GD converges a way slower than L-BFGS. To achieve 1% difference,
    // it requires 90 iterations in GD. No matter how hard we increase
    // the number of iterations in GD here, the lossGD will be always
    // larger than lossLBFGS. This is based on observation, no theoretically guaranteed
    val edate = new Date
    System.err.println(s"Finished test at ${edate.toString}: duration=${((edate.getTime-sdate.getTime)/1000).toInt}")
    assert(Math.abs((lossGD.last - loss.last) / loss.last) < 0.02,
      "LBFGS should match GD result within 2% difference.")

  }
}
