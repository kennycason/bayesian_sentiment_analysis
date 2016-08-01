package com.kennycason.ml.classifier.bayes.performance

import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
import org.slf4j.LoggerFactory

/**
 * Created by kenny on 6/29/16.
 *
 * This class's goal is to make the model smaller without compromising accuracy
 */
class ModelPruner(val pruneThreshold: Float = 0.05f,
                  val minFrequency: Int = 2) {
    private val logger = LoggerFactory.getLogger(ModelPruner::class.java)

    fun prune(classifier: NaiveBayesianClassifier) {
        logger.info("model size before pruning: ${classifier.model.size}")
        val iterator = classifier.model.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.value.positiveCount + entry.value.negativeCount < minFrequency
                    || Math.abs(entry.value.positiveProbability - .5) < pruneThreshold) {
                iterator.remove()
            }
        }
        logger.info("model size after pruning: ${classifier.model.size}")
    }

    fun prune(classifier: StochasticBayesianClassifier) {
        classifier.classifiers.forEach { classifier -> prune(classifier) }
    }

}