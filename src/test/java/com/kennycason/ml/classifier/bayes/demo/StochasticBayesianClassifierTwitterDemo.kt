package com.kennycason.ml.classifier.bayes.demo

import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.ml.classifier.bayes.helper.BayesianClassifierResultEvaluator
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
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
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/17/16.
 *
 * Some simple Twitter tests based off the twitter sentiment analysis competition and a sample of hand rated Twitter data.
 * https://inclass.kaggle.com/c/si650winter11
 * Current top record is 94.3%
 */
fun main(args: Array<String>) {
    val classifierDemo = StochasticBayesianClassifierTwitterDemo()
    classifierDemo.stochasticClassifierNGramKaggleData()
    classifierDemo.stochasticClassifierNGramHandRatedData()
}

class StochasticBayesianClassifierTwitterDemo {
    private val logger = LoggerFactory.getLogger(StochasticBayesianClassifierTwitterDemo::class.java)
    private val resultEvaluator = BayesianClassifierResultEvaluator()
    private val resourcePath = "com/kennycason/ml/classifier/bayes/corpus/twitter/"
    private val textTokenizer = EnglishTokenizer()
    private val errorDelta = 0.2f

    // TODO there are 30k unlabeled test data to get manually rated
    // Train data learned to 99.95%
    fun stochasticClassifierNGramKaggleData() {
        val classifier = StochasticBayesianClassifier(classifierCount = 15, samplingPercent = 0.5f)
        val resourceLoader = ResourceLoader()
        val tokenizer = NGramTokenizer<String>(2)

        val positiveGrams = buildGrams(resourceLoader.toLines(resourcePath + "kaggle/twitter_train_pos.txt"), tokenizer)
        val negativeGrams = buildGrams(resourceLoader.toLines(resourcePath + "kaggle/twitter_train_neg.txt"), tokenizer)

        val start = System.currentTimeMillis()
        classifier.trainPositive(positiveGrams)
        classifier.trainNegative(negativeGrams)
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, positiveGrams, 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, negativeGrams, 0.0f, errorDelta)
    }

    // using hand rated twitter data
    fun stochasticClassifierNGramHandRatedData() {
        val classifier = StochasticBayesianClassifier(classifierCount = 15, samplingPercent = 0.5f)
        val resourceLoader = ResourceLoader()
        val tokenizer = NGramTokenizer<String>(2)

        val positiveGrams = buildGrams(resourceLoader.toLines(resourcePath + "manual_5000_pos.txt"), tokenizer)
        val negativeGrams = buildGrams(resourceLoader.toLines(resourcePath + "manual_5000_neg.txt"), tokenizer)

        // split the data between test/train
        val positiveTrain = positiveGrams.subList(0, positiveGrams.size / 2)
        val negativeTrain = negativeGrams.subList(0, negativeGrams.size / 2)
        val positiveTest = positiveGrams.subList(positiveGrams.size / 2, positiveGrams.size)
        val negativeTest = negativeGrams.subList(negativeGrams.size / 2, negativeGrams.size)

        val start = System.currentTimeMillis()
        classifier.trainPositive(positiveTrain)
        classifier.trainNegative(negativeTrain)
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, positiveTrain, 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, negativeTrain, 0.0f, errorDelta)

        logger.info("Test Data Results")
        val positiveResult = resultEvaluator.evaluate(classifier, positiveTest, 1.0f, errorDelta)
        val negativeResult = resultEvaluator.evaluate(classifier, negativeTest, 0.0f, errorDelta)
        val percentRated = (positiveResult.percentRated() + negativeResult.percentRated()) / 2.0f
        val percentAccuracy = (positiveResult.percentCorrect() + negativeResult.percentCorrect()) / 2.0f
        logger.info("$errorDelta\t$percentRated\t$percentAccuracy")
    }

    private fun buildGrams(corpus: List<String>, gramTokenizer: GramTokenizer<String>) =
            corpus.map { data ->
                gramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize(data))) }

}