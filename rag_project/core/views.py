from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.db.models import Q
from django.views.generic import TemplateView
from django.views.generic.edit import CreateView,UpdateView,DeleteView
from django.http import JsonResponse
import requests

from core.ai_engine.document_reader import DocumentReader

import os
import json
import faiss
import numpy as np

from django.conf import settings
from openai import OpenAI

class HomeView_main(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        ai_folder = os.path.join(settings.BASE_DIR, "AI_Data")

        if os.path.exists(ai_folder):
            files = os.listdir(ai_folder)
        else:
            files = []

        context["files"] = files

        return context


class HomeView(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        files = os.listdir("AI_Data")

        reader = DocumentReader()
        documents = reader.load_documents()

        context["files"] = files
        context["documents"] = documents

        return context
