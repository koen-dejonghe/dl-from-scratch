package botkop.data

import botkop.{numsca => ns}
import ns.Tensor
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable
import scorch.data.loader.DataLoader

import scala.io.Source
import scala.language.postfixOps
import scala.util.Random

class FashionMnistDataLoader(val mode: String,
                             miniBatchSize: Int,
                             take: Option[Int] = None,
                             seed: Long = 231)
    extends DataLoader
    with LazyLogging {

  Random.setSeed(seed)

  val file: String = mode match {
    case "train" => "data/fashionmnist/fashion-mnist_train.csv"
    case "test"  => "data/fashionmnist/fashion-mnist_test.csv"
  }

  def getLines: List[String] = {
    val src = Source.fromFile(file)
    val lines = src.getLines().toList
    src.close()
    Random.shuffle(lines.tail) // skip header
  }

  val numFeatures = 784
  val numEntries: Int = getLines.length

  override val numSamples: Int = take match {
    case Some(n) => math.min(n, numEntries)
    case None    => numEntries
  }

  override val numBatches: Int =
    (numSamples / miniBatchSize) +
      (if (numSamples % miniBatchSize == 0) 0 else 1)

  // this should fit in memory
  val data: Seq[(Variable, Variable)] = getLines
    .take(take.getOrElse(numSamples))
    .sliding(miniBatchSize, miniBatchSize)
    .toSeq
    .map { lines =>
      val (xData, yData) = lines
        .foldLeft(List.empty[Float], List.empty[Float]) {
          case ((xs, ys), line) =>
            val tokens = line.split(",")
            val (y, x) =
              (tokens.head.toFloat, tokens.tail.map(_.toFloat / 255).toList)
            (x ::: xs, y :: ys)
        }

      val x = Variable(Tensor(xData.toArray).reshape(yData.length, numFeatures))
      val y = Variable(Tensor(yData.toArray).reshape(yData.length, 1))
      (x, y)
    }

  lazy val meanImage: Tensor = {
    val m = ns.zeros(1, numFeatures)
    data.foreach {
      case (x, _) =>
        m += ns.sum(x.data, axis = 0)
    }
    m /= numSamples
    m
  }

  if (mode == "train") normalize(meanImage)

  def normalize(meanImage: Tensor): Unit =
    data.foreach {
      case (x, _) =>
        x.data -= meanImage
    }

  override def iterator: Iterator[(Variable, Variable)] =
    Random.shuffle(data.toIterator)

}
