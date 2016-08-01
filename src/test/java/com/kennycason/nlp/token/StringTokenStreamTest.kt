package com.kennycason.nlp.token

import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import org.junit.Test
import kotlin.test.assertEquals

/**
 * Created by kenny on 6/14/16.
 */
class StringTokenStreamTest {
    private val textTokenizer = EnglishTokenizer()

    @Test
    fun basicTest() {
        val tokenStream = StringTokenStream(textTokenizer.tokenize("1 2 3 4 5"))
        assertEquals(5, tokenStream.size())
        assertEquals("1", tokenStream.get(0))

        tokenStream.forEachIndexed { i, token ->
            assertEquals(tokenStream.get(i), token)
        }

        val window = tokenStream.window(1, 4)
        assertEquals(3, window.size)
        assertEquals("2", window.get(0))
        assertEquals("3", window.get(1))
        assertEquals("4", window.get(2))
    }

}