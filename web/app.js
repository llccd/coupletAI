var app = new Vue({
  el: '#app',
  data: {
    message: '',
    timeused: 0,
    topk: 1,
    result: ''
  },
  methods: {
    getCouplet: function () {
    if(!this.message || !this.topk) return;
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: this.message, topk: this.topk })
    };
    fetch("/api/v1/couplet", requestOptions)
      .then(response => response.json())
      .then(data => {this.result = this.message+'<br>'+data.result; this.timeused = data.timeused});
    }
  }
})