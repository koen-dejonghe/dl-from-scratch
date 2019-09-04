package botkop.data

import botkop.numsca.Tensor
import botkop.{numsca => ns}
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

  val lines: List[String] = {
    logger.info(s"starting to read $file")
    val src = Source.fromFile(file)
    val lines = src.getLines().toList
    src.close()
    logger.info(s"reading $file: done")
    Random.shuffle(lines.tail) // skip header
  }

  val numFeatures = 784
  val numEntries: Int = lines.length

  override val numSamples: Int = take match {
    case Some(n) => math.min(n, numEntries)
    case None    => numEntries
  }

  override val numBatches: Int =
    (numSamples / miniBatchSize) +
      (if (numSamples % miniBatchSize == 0) 0 else 1)

//  val data: Seq[(Variable, Variable)] = lines
//    .take(take.getOrElse(numSamples))
//    .sliding(miniBatchSize, miniBatchSize)
//    .map { lines =>
//      val (xData, yData) = lines
//        .foldLeft(List.empty[Float], List.empty[Float]) {
//          case ((xs, ys), line) =>
//            val tokens = line.split(",")
//            val (y, x) =
//              (tokens.head.toFloat, tokens.tail.map(_.toFloat / 255).toList)
//            (x ::: xs, y :: ys)
//        }
//      val x = Variable(Tensor(xData.toArray).reshape(yData.length, numFeatures))
//      val y = Variable(Tensor(yData.toArray).reshape(yData.length, 1))
//      (x, y)
//    }
//    .toSeq

  val data: Seq[(Variable, Variable)] = lines
    .take(take.getOrElse(numSamples))
    .sliding(miniBatchSize, miniBatchSize)
    .map { lines =>
      val batchSize = lines.length

      val xs = Array.fill[Float](numFeatures * batchSize)(elem = 0)
      val ys = Array.fill[Float](batchSize)(elem = 0)

      lines.zipWithIndex.foreach { case (line, lnr) =>
        val tokens = line.split(",")
        ys(lnr) = tokens.head.toFloat
        tokens.tail.zipWithIndex.foreach { case (sx, i) =>
          xs(lnr * numFeatures + i) = sx.toFloat / 255
        }
      }

      val x = Variable(Tensor(xs).reshape(batchSize, numFeatures))
      val y = Variable(Tensor(ys).reshape(batchSize, 1))
      (x, y)
    }
    .toSeq

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
