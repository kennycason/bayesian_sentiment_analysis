package com.kennycason.ml.classifier.bayes

import com.kennycason.nlp.gram.Gram
import com.kennycason.nlp.util.RandomSampler

/**
 * Created by kenny on 6/17/16.
 *
 * Similar to what Random Forest is to Decision Tree, create a set of Bayesian Classifiers where
 * each classifier is trained on a random subset of the corpus.
 */
class StochasticBayesianClassifier(override val exclusions: MutableSet<String> = mutableSetOf(),
                                   override val interestingGramsCount: Int = 15,
                                   override val assumePrioriWhenSubjectAbsent: Boolean = false,
                                   override val negativeProbabilityPriori: Float = 0.4f,
                                   val samplingPercent: Float = 0.2f,
                                   val classifierCount: Int = 5,
                                   val preBuiltClassifiers: MutableList<NaiveBayesianClassifier> = mutableListOf()) : BayesianClassifier {
    private val randomSampler = RandomSampler()
    val classifiers: List<NaiveBayesianClassifier> = when (preBuiltClassifiers.isEmpty()) {
        true -> (1..classifierCount).map { NaiveBayesianClassifier(
                exclusions = exclusions,
                interestingGramsCount = interestingGramsCount,
                assumePrioriWhenSubjectAbsent = assumePrioriWhenSubjectAbsent,
                negativeProbabilityPriori = negativeProbabilityPriori) }.toList()
        false -> preBuiltClassifiers
    }

    override fun classify(grams: List<Gram<String>>) = classifiers
            .map { classifier -> classifier.classify(grams) }
            .average()
            .toFloat()

    fun finalizeTraining() = classifiers.forEach { classifier -> classifier.finalizeTraining() }

    fun trainNegative(corpus: List<List<Gram<String>>>) {
        val sampleSize = (samplingPercent * corpus.size).toInt()
        classifiers.forEach { classifier ->
            randomSampler.sample(corpus, sampleSize).forEach { sampledCorpus ->
                classifier.trainNegative(sampledCorpus)
            }
        }
    }

    fun trainPositive(corpus: List<List<Gram<String>>>) {
        val sampleSize = (samplingPercent * corpus.size).toInt()
        classifiers.forEach { classifier ->
            randomSampler.sample(corpus, sampleSize).forEach { sampledCorpus ->
                classifier.trainPositive(sampledCorpus)
            }
        }
    }

}
