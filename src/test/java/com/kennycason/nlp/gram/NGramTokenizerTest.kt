package com.kennycason.nlp.gram

import com.kennycason.nlp.token.tokenizer.WhiteSpaceTokenizer
import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.expect

/**
 * Created by kenny on 6/14/16.
 */
class NGramTokenizerTest {
    private val textTokenizer = EnglishTokenizer()

    @Test(expected = IllegalArgumentException::class)
    fun zerogram() {
        NGramTokenizer<String>(0)
    }

    @Test
    fun unigram() {
        val unigramTokenizer = NGramTokenizer<String>(1)
        val unigrams = unigramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("I love Kotlin")))
        assertEquals(3, unigrams.size)
        assertEquals("i", unigrams.get(0).buildToken())
        assertEquals("love", unigrams.get(1).buildToken())
        assertEquals("kotlin", unigrams.get(2).buildToken())
    }

    @Test
    fun bigram() {
        val bigramTokenizer = NGramTokenizer<String>(2)
        val bigrams = bigramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("I love Kotlin")))
        assertEquals(2, bigrams.size)
        assertEquals("i_love", bigrams.get(0).buildToken())
        assertEquals("love_kotlin", bigrams.get(1).buildToken())
    }

    @Test
    fun trigram() {
        val trigramTokenizer = NGramTokenizer<String>(3)
        val trigrams = trigramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("I love Kotlin")))
        assertEquals(1, trigrams.size)
        assertEquals("i_love_kotlin", trigrams.get(0).buildToken())
    }

}