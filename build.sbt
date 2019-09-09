import Dependencies._

ThisBuild / scalaVersion     := "2.12.9"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "be.botkop"
ThisBuild / organizationName := "botkop"

lazy val root = (project in file("."))
  .settings(
    name := "dl-from-scratch",
    libraryDependencies += scalaTest % Test,
    libraryDependencies += "be.botkop" %% "numsca" % "0.1.5"

  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
