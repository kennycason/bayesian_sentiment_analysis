package com.kennycason.ml.classifier.bayes.performance

import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.nlp.gram.Gram
import org.slf4j.LoggerFactory

/**
 * Created by kenny on 6/29/16.
 */
class ClassifierSimulator(val errorDelta: Float) {
    private val logger = LoggerFactory.getLogger(ClassifierSimulator::class.java)

    fun reportSimulationResults(simulations: List<SimulationResult>) {
        var squaredDiffFrom50 = 0.0
        var error = 0.0
        println("positive actual\tpercent positive actual\tpercent positive classifier")
        simulations.forEach { simulation ->
            val percentPositiveActual = simulation.positiveActual.toFloat() / (simulation.positiveActual + simulation.negativeActual)
            val percentPositiveClassifier = simulation.positiveClassifier.toFloat() / (simulation.positiveClassifier + simulation.negativeClassifier)
            println("${simulation.positiveActual}\t$percentPositiveActual\t$percentPositiveClassifier")
            squaredDiffFrom50 += (percentPositiveClassifier - .5) * (percentPositiveClassifier - .5)
            error += Math.abs(percentPositiveClassifier - 0.5)
        }
        logger.info("squared error ${Math.sqrt(squaredDiffFrom50)}")
        logger.info("avg error ${squaredDiffFrom50 / simulations.size}")
    }

    fun runSimulation(classifier: BayesianClassifier,
                      sample: Pair<List<List<Gram<String>>>, List<List<Gram<String>>>>): SimulationResult {
        var positiveClassifier = 0
        var negativeClassifier = 0
        sample.first.forEach { sample ->
            val positiveProbability = classifier.classify(sample)
            if (positiveProbability >= 1.0f - errorDelta) { positiveClassifier++ }
            else if (positiveProbability <= errorDelta) { negativeClassifier++ }
        }
        sample.second.forEach { sample ->
            val positiveProbability = classifier.classify(sample)
            if (positiveProbability >= 1.0f - errorDelta) { positiveClassifier++ }
            else if (positiveProbability <= errorDelta) { negativeClassifier++ }
        }
        return SimulationResult(
                nActual = sample.first.size + sample.second.size,
                nClassifier = positiveClassifier + negativeClassifier,
                positiveActual = sample.first.size,
                negativeActual = sample.second.size,
                positiveClassifier = positiveClassifier,
                negativeClassifier = negativeClassifier)
    }

}

data class SimulationResult(val nActual: Int,
                            val nClassifier: Int,
                            val positiveActual: Int,
                            val negativeActual: Int,
                            val positiveClassifier: Int,
                            val negativeClassifier: Int)