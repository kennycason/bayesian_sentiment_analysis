package com.kennycason.ml.classifier.bayes.demo

import com.kennycason.ml.classifier.bayes.helper.BayesianClassifierResultEvaluator
import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
import com.kennycason.ml.classifier.bayes.model.StreamingModelPersister
import com.kennycason.ml.classifier.bayes.performance.ModelPruner
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
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/17/16.
 *
 * Demonstration heap usage and classifier speed (posts per min)
 */
fun main(args: Array<String>) {
    val classifierDemo = BayesianClassifierPerformanceDemo()
   // classifierDemo.loadModelWatchHeap()
    classifierDemo.processReviewSpeed()
}

class BayesianClassifierPerformanceDemo() {
    private val logger = LoggerFactory.getLogger(BayesianClassifierPerformanceDemo::class.java)
    private val resultEvaluator = BayesianClassifierResultEvaluator()
    private val textTokenizer = EnglishTokenizer()

    fun processReviewSpeed() {
        val imdbCorpus = ImdbLoader().loadFromResourceDirectory("com/kennycason/ml/classifier/bayes/corpus/imdb_sample/")
        val twitter = ResourceLoader().toLines("com/kennycason/ml/classifier/bayes/corpus/twitter/manual_5000_pos.txt")
        val tokenizer = NGramTokenizer<String>(2)
        val classifier = StreamingModelPersister().load(File("/tmp/pruned_model.json"))
        //val classifier = StreamingModelPersister().load(File("/tmp/imdb_stochastic_bayesian_classifier.json"))

        val experimentTime = 60000 // 60 seconds
        val startTime = System.currentTimeMillis()
        var rated = 0
        while(true) {
            for (review in imdbCorpus.test.positive) {
                classifier.classify(buildGrams(review, tokenizer))
                rated++
                if (System.currentTimeMillis() - startTime > experimentTime) {
                    println("$rated posts/min")
                    return
                }
            }
        }
    }

    // watch heap in jvisualvm
    fun loadModelWatchHeap() {
        //val classifier = StreamingModelPersister().load(File("/tmp/pruned_model.json")) // 335mb
        val classifier = StreamingModelPersister().load(File("/tmp/imdb_stochastic_bayesian_classifier.json")) // 888mb
        Thread.sleep(10000)
    }

    private fun buildGrams(data: String, gramTokenizer: GramTokenizer<String>) =
                   gramTokenizer.tokenize(
                           StringTokenStream(textTokenizer.tokenize(data)))


}