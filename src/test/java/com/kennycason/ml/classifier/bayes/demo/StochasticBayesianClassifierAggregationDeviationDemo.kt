package com.kennycason.ml.classifier.bayes.demo

import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.ml.classifier.bayes.helper.BayesianClassifierResultEvaluator
import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
import com.kennycason.ml.classifier.bayes.model.StreamingModelPersister
import com.kennycason.ml.classifier.bayes.performance.ClassifierSimulator
import com.kennycason.nlp.data.imdb.ImdbDataCorpus
import com.kennycason.nlp.data.imdb.ImdbLoader
import com.kennycason.nlp.gram.Gram
import com.kennycason.nlp.gram.GramTokenizer
import com.kennycason.nlp.gram.NGramTokenizer
import com.kennycason.nlp.gram.SkipGramTokenizer
import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import com.kennycason.nlp.util.RandomSampler
import com.kennycason.nlp.util.ResourceLoader
import org.eclipse.collections.api.RichIterable
import org.eclipse.collections.impl.factory.Lists
import org.eclipse.collections.impl.list.mutable.ListAdapter
import org.junit.Test
import org.slf4j.LoggerFactory
import java.io.File
import java.util.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/17/16.
 *
 * Perform aggregation simulations to ensure aggregations are representative of actual aggregations.
 * The idea is that while we achieve 92-95% accuracy in single tests we want the aggregation accuracy to be higher.
 *
 * This test is on the full Imdb review set and must be externally downloaded to run this simulation.
 */
fun main(args: Array<String>) {
    if (args.size == 0) {
        // example "/Users/kenny/Downloads/aclImdb/"
        throw IllegalArgumentException("First parameter must be input directory.")
    }
    val classifierDemo = StochasticBayesianClassifierAggregationDeviationDemo(args[0])
    classifierDemo.runImdb()
  //  classifierDemo.runTwitter()
}

class StochasticBayesianClassifierAggregationDeviationDemo(val imdbInputDirectory: String) {
    private val logger = LoggerFactory.getLogger(BayesianClassifierImdbDemo::class.java)
    private val random = Random()
    private val randomSampler = RandomSampler()
    private val resultEvaluator = BayesianClassifierResultEvaluator()
    private val textTokenizer = EnglishTokenizer()
    private val ngramTokenizer = NGramTokenizer<String>(2)
    private val errorDelta = 0.2f
    private val imdbDataCorpus = ImdbLoader().load(imdbInputDirectory)

    fun runImdb() {
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.2f)
        val simulator = ClassifierSimulator(errorDelta)

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, ngramTokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, ngramTokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Beginning simulation")
        simulator.reportSimulationResults((1..100).map { simulator.runSimulation(classifier, subSampleReviews(imdbDataCorpus)) })
        logger.info("Ending simulation")
    }

    fun runTwitter() {
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.2f)
        val resourcePath = "com/kennycason/ml/classifier/bayes/corpus/twitter/"
        val resourceLoader = ResourceLoader()
        val positiveGrams = buildGrams(resourceLoader.toLines(resourcePath + "manual_5000_pos.txt"), ngramTokenizer)
        val negativeGrams = buildGrams(resourceLoader.toLines(resourcePath + "manual_5000_neg.txt"), ngramTokenizer)
        val simulator = ClassifierSimulator(errorDelta)

        val start = System.currentTimeMillis()
        classifier.trainPositive(positiveGrams)
        classifier.trainNegative(negativeGrams)
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Beginning simulation")
        simulator.reportSimulationResults((1..100).map { simulator.runSimulation(classifier, subSampleTweets(positiveGrams, negativeGrams)) })
        logger.info("Ending simulation")
    }

    private fun subSampleReviews(imdbDataCorpus: ImdbDataCorpus): Pair<List<List<Gram<String>>>, List<List<Gram<String>>>> {
        return Pair(
                buildGrams(randomSampler.sample(imdbDataCorpus.test.positive, 1000), ngramTokenizer),
                buildGrams(randomSampler.sample(imdbDataCorpus.test.negative, 1000), ngramTokenizer))
    }

    private fun subSampleTweets(positiveGrams: List<List<Gram<String>>>, negativeGrams: List<List<Gram<String>>>): Pair<List<List<Gram<String>>>, List<List<Gram<String>>>> {
        return Pair(
                randomSampler.sample(positiveGrams, 1000),
                randomSampler.sample(negativeGrams, 1000))
    }

    private fun buildGrams(corpus: List<String>, gramTokenizer: GramTokenizer<String>) =
            corpus.map { data ->
                gramTokenizer.tokenize(
                        StringTokenStream(textTokenizer.tokenize(data))) }

}