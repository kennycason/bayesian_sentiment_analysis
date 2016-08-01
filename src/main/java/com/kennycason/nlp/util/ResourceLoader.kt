package com.kennycason.nlp.util;

import org.apache.commons.io.IOUtils
import java.io.InputStream


class ResourceLoader {

    fun toString(resource: String): String {
        return IOUtils.toString(toInputStream(resource));
    }

    fun toLines(resource: String): List<String> {
        return IOUtils.readLines(toInputStream(resource));
    }

    fun toInputStream(resource: String): InputStream {
        return Thread.currentThread().getContextClassLoader().getResourceAsStream(resource);
    }

}