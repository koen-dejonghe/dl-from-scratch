package botkop.data

import botkop.autograd.Variable
import botkop.numsca.Tensor

trait DataLoader extends scala.collection.immutable.Iterable[(Variable, Variable)] {
  def numSamples: Int
  def numBatches: Int
  def mode: String
  def iterator: Iterator[(Variable, Variable)]
}

@SerialVersionUID(123L)
case class YX(y: Float, x: Array[Float]) extends Serializable

