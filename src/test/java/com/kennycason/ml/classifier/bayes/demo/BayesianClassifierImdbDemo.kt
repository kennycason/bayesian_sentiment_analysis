package com.kennycason.ml.classifier.bayes.demo

import com.kennycason.ml.classifier.bayes.helper.BayesianClassifierResultEvaluator
import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
import com.kennycason.ml.classifier.bayes.model.StreamingModelPersister
import com.kennycason.ml.classifier.bayes.performance.ModelPruner
import com.kennycason.nlp.data.imdb.ImdbDataCorpus
import com.kennycason.nlp.data.imdb.ImdbLoader
import com.kennycason.nlp.gram.*
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
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/17/16.
 *
 * A demo using full Imdb movie reviews. http://ai.stanford.edu/~amaas/data/sentiment/
 * The data set is large so must be downloaded to local computer.
 *
 * Refer to machine_learning/sentiment_analysis.md for more information.
 */
fun main(args: Array<String>) {
    if (args.size == 0) {
        // example "/Users/kenny/Downloads/aclImdb/"
        throw IllegalArgumentException("First parameter must be input directory.")
    }
    val classifierDemo = BayesianClassifierImdbDemo(args[0])
    // comment out/in the specific demo
   // classifierDemo.stochasticClassifierNGramAccuracyVsRatedPercentage()
    //classifierDemo.loadModel()
  //  classifierDemo.prune()
    classifierDemo.stochasticClassifierNGram()
//    classifierDemo.stochasticClassifierSkipGram()
//    classifierDemo.bayesianClassifierNGram()
//    classifierDemo.bayesianClassifierSkipGram()
}

class BayesianClassifierImdbDemo(val imdbInputDirectory: String) {
    private val logger = LoggerFactory.getLogger(BayesianClassifierImdbDemo::class.java)
    private val resultEvaluator = BayesianClassifierResultEvaluator()
    private val textTokenizer = EnglishTokenizer()
    private val errorDelta = 0.2f
    private val imdbDataCorpus = ImdbLoader().load(imdbInputDirectory)

    fun stochasticClassifierNGram() {
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.2f)
        val ngramTokenizer = NGramTokenizer<String>(2)

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, ngramTokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, ngramTokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.negative, ngramTokenizer), 0.0f, errorDelta)

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)

//        val modelPersister = StreamingModelPersister()
//        modelPersister.persist(classifier, File("/tmp/imdb_stochastic_bayesian_classifier.json"))
    }

    fun prune() {
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.2f)
        val ngramTokenizer = NGramTokenizer<String>(2)
        val modelPruner = ModelPruner()

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, ngramTokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, ngramTokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)

        logger.info("Pruned Test Data Results")
        modelPruner.prune(classifier)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)
        StreamingModelPersister().persist(classifier, File("/tmp/pruned_model.json"))
    }

    fun compositeBiGramAndTriGram() {
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.2f)
        val ngramTokenizer = CompositeGramTokenizer<String>(Lists.mutable.of(NGramTokenizer<String>(2), NGramTokenizer<String>(3)))
        val modelPruner = ModelPruner()

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, ngramTokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, ngramTokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)

        logger.info("Pruned Test Data Results")
        modelPruner.prune(classifier)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)
        StreamingModelPersister().persist(classifier, File("/tmp/pruned_model.json"))
    }

    // iterate over error thresholds and measure accuracy vs comments rated
    fun stochasticClassifierNGramAccuracyVsRatedPercentage() {
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.2f)
        val ngramTokenizer = NGramTokenizer<String>(2)

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, ngramTokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, ngramTokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Test Data Results")
        logger.info("error threshold, percent comments rated, percent accuracy")
        resultEvaluator.print = false
        for (i in (1..50)) {
            val errorDelta = i * 0.01f
            val positiveResult = resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
            val negativeResult = resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)
            val percentRated = (positiveResult.percentRated() + negativeResult.percentRated()) / 2.0f
            val percentAccuracy = (positiveResult.percentCorrect() + negativeResult.percentCorrect()) / 2.0f
            println("$errorDelta\t$percentRated\t$percentAccuracy")
        }
        resultEvaluator.print = true
    }

    // a pretty large model
    fun loadModel() {
        val ngramTokenizer = NGramTokenizer<String>(2)
        val modelPersister = StreamingModelPersister()
        logger.info("Loading classifier")
        val classifier = modelPersister.load(File("/tmp/imdb_stochastic_bayesian_classifier.json"))
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)
    }


    fun stochasticClassifierSkipGram() {
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.2f)
        val skipgramTokenizer = SkipGramTokenizer<String>(2, 2)

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, skipgramTokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, skipgramTokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.positive, skipgramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.negative, skipgramTokenizer), 0.0f, errorDelta)

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, skipgramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, skipgramTokenizer), 0.0f, errorDelta)
    }

    fun bayesianClassifierNGram() {
        val classifier = NaiveBayesianClassifier()
        val ngramTokenizer = NGramTokenizer<String>(2)

        val start = System.currentTimeMillis()
        buildGrams(imdbDataCorpus.train.positive, ngramTokenizer).forEach { data -> classifier.trainPositive(data) }
        buildGrams(imdbDataCorpus.train.negative, ngramTokenizer).forEach { data -> classifier.trainNegative(data) }
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.negative, ngramTokenizer), 0.0f, errorDelta)

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, ngramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, ngramTokenizer), 0.0f, errorDelta)
    }

    fun bayesianClassifierSkipGram() {
        val classifier = NaiveBayesianClassifier()
        val skipgramTokenizer = SkipGramTokenizer<String>(2, 2)

        val start = System.currentTimeMillis()
        buildGrams(imdbDataCorpus.train.positive, skipgramTokenizer).forEach { data -> classifier.trainPositive(data) }
        buildGrams(imdbDataCorpus.train.negative, skipgramTokenizer).forEach { data -> classifier.trainNegative(data) }
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.positive, skipgramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.negative, skipgramTokenizer), 0.0f, errorDelta)

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, skipgramTokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, skipgramTokenizer), 0.0f, errorDelta)
    }
    
    private fun buildGrams(corpus: List<String>, gramTokenizer: GramTokenizer<String>) =
            corpus.map { data ->
                gramTokenizer.tokenize(
                        StringTokenStream(textTokenizer.tokenize(data))) }

}