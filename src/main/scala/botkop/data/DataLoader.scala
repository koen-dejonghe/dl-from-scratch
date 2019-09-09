package botkop.data

import botkop.autograd.Variable

trait DataLoader extends scala.collection.immutable.Iterable[(Variable, Variable)] {
  def numSamples: Int
  def numBatches: Int
  def mode: String
}

@SerialVersionUID(123L)
case class YX(y: Float, x: Array[Float]) extends Serializable

