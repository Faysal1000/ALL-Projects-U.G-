using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Notification
    {
        [Key]
        public int NotificationId { get; set; }

        [Required]
        public string RecipientType { get; set; }

        [Required]
        public int RecipientId { get; set; }

        [Required]
        public string Message { get; set; }

        public DateTime SentAt { get; set; } = DateTime.Now;

        public bool Read { get; set; } = false;
    }
}