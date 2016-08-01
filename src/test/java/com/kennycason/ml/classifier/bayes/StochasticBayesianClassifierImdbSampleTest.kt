package com.kennycason.ml.classifier.bayes

import com.kennycason.ml.classifier.bayes.helper.BayesianClassifierResultEvaluator
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
import org.eclipse.collections.impl.list.mutable.ListAdapter
import org.junit.Test
import org.slf4j.LoggerFactory
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/17/16.
 *
 * Some simple sanity tests
 */
class StochasticBayesianClassifierImdbSampleTest {
    private val logger = LoggerFactory.getLogger(StochasticBayesianClassifierImdbSampleTest::class.java)
    private val resultEvaluator = BayesianClassifierResultEvaluator()
    private val resourcePath = "com/kennycason/ml/classifier/bayes/corpus/imdb_sample/"
    private val textTokenizer = EnglishTokenizer()
    private val errorDelta = 0.2f

    @Test
    fun stochasticClassifierNGram() {
        logger.info("ngram classifier")
        val classifier = StochasticBayesianClassifier(classifierCount = 15, samplingPercent = 0.5f)
        val imdbDataCorpus = ImdbLoader().loadFromResourceDirectory(resourcePath)
        val tokenizer = NGramTokenizer<String>(2)

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, tokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, tokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.positive, tokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.negative, tokenizer), 0.0f, errorDelta)

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, tokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, tokenizer), 0.0f, errorDelta)
    }

    @Test
    fun stochasticClassifierSkipGram() {
        logger.info("skipgram classifier")
        val classifier = StochasticBayesianClassifier(classifierCount = 10, samplingPercent = 0.3f)
        val imdbDataCorpus = ImdbLoader().loadFromResourceDirectory(resourcePath)
        val tokenizer = SkipGramTokenizer<String>(2, 2)

        val start = System.currentTimeMillis()
        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive, tokenizer))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative, tokenizer))
        classifier.finalizeTraining()
        logger.info("${System.currentTimeMillis() - start} ms to train")

        logger.info("Training Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.positive, tokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.train.negative, tokenizer), 0.0f, errorDelta)

        logger.info("Test Data Results")
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.positive, tokenizer), 1.0f, errorDelta)
        resultEvaluator.evaluate(classifier, buildGrams(imdbDataCorpus.test.negative, tokenizer), 0.0f, errorDelta)
    }

    private fun buildGrams(corpus: List<String>, gramTokenizer: GramTokenizer<String>) =
            corpus.map { data ->
                gramTokenizer.tokenize(
                        StringTokenStream(textTokenizer.tokenize(data))) }


}