package botkop.linreg

import botkop.{numsca => ns}
import ns._
import scorch.autograd.Variable

import scala.util.Random

object LinearRegression extends App {

  val numFeatures = 2
  val numExamples = 1000

  val trueW = Tensor(2, -3.4).reshape(2, 1)
  val trueB = Tensor(4.2)

  val features = ns.randn(numExamples, numFeatures)
  println(features.shape.toList)

  val labels = (features dot trueW) + trueB
  val sigma = ns.randn(labels.shape) * 0.01
  labels += sigma
  println(labels.shape.toList)

  def dataIter(batchSize: Int, features: Tensor, labels: Tensor): Iterator[(Variable, Variable)] = {
    val numExamples = features.shape(0)
    val indices = Random.shuffle((0 until numExamples).toList)

    def extract(t: Tensor, indices: Seq[Int]): Variable = Variable(ns.concatenate(indices.map(i => t(i))))

    indices.sliding(batchSize, batchSize).map { l =>
      (extract(features, l), extract(labels.T, l))
    }
  }

  val batchSize = 10
//  dataIter(batchSize, features, labels).foreach { case (f, l) =>
//    println(f)
//    println(l)
//    println
//  }

  val w = Variable(ns.randn(numFeatures, 1) * 0.01)
  val b = Variable(ns.zeros(1))

  def linReg(x: Variable, w: Variable, b: Variable): Variable = (x dot w) + b

  def squaredLoss(yHat: Variable, y: Variable): Variable = (yHat - y) ** 2 / 2

  def sgd(params: Seq[Variable], lr: Double, batchSize: Int): Unit =
    params.foreach { p =>
      p.data -= lr * p.grad.data / batchSize
      p.grad.data := 0.0
    }

  val lr = 0.03
  val numEpochs = 10
  val net: (Variable, Variable, Variable) => Variable = linReg
  val loss: (Variable, Variable) => Variable = squaredLoss

  (0 until numEpochs).foreach { epoch =>
    dataIter(batchSize, features, labels).foreach { case (x, y) =>
        val l = loss(net(x, w, b), y).mean()
        l.backward()
        sgd(Seq(w, b), lr, batchSize)
    }
    val l = loss(net(Variable(features), w, b), Variable(labels))
    print(s"epoch: $epoch, loss ${l.mean()}\n")
  }



}
