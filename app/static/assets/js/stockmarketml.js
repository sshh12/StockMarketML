const socket = new WebSocket(window.location.href.replace("http", "ws") + "prediction");

socket.addEventListener('open', (event) => {
    console.log('Connected.');
});

socket.addEventListener('message', (event) => {
    let msg = JSON.parse(event.data);
    if(msg.status == "loading") {
      $(`#${msg.stock}-data`).html(`<div class=\"small text-muted\">Please wait...</div><div>Analyzing</div`)
    } else if(msg.status == "complete") {
      $(`#${msg.stock}-data`).html(`<div class=\"small text-muted\">${msg.numheadlines} Headlines Analyzed</div><div>Just Now</div`)
      let pred = msg.prediction;
      if(pred[0] > pred[1]) {
        $(`#${msg.stock}-circle`).html(`<div class="mx-auto chart-circle chart-circle-xs" data-value="${pred[0]}" data-thickness="3" data-color="green"><div class="chart-circle-value">${ (pred[0] * 100).toFixed(0) }%</div></div>`)
      } else {
        $(`#${msg.stock}-circle`).html(`<div class="mx-auto chart-circle chart-circle-xs" data-value="${pred[1]}" data-thickness="3" data-color="red"><div class="chart-circle-value">${ (pred[1] * 100).toFixed(0) }%</div></div>`)
      }
      updateCircles();
    }
});

function queryPrediction(stock) {
  socket.send(stock);
}

function updateCircles() {
  if ($('.chart-circle').length) {
    require(['circle-progress'], function() {
      $('.chart-circle').each(function() {
        let $this = $(this);

        $this.circleProgress({
          fill: {
            color: tabler.colors[$this.attr('data-color')] || tabler.colors.blue
          },
          size: $this.height(),
          startAngle: -Math.PI / 4 * 2,
          emptyFill: '#F4F4F4',
          lineCap: 'round'
        });
      });
    });
  }
}
