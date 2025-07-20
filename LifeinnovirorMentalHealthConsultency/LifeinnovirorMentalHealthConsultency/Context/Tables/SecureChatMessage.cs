using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class SecureChatMessage
    {
        [Key]
        public int MessageId { get; set; }

        public int FromUserId { get; set; }

        public int ToUserId { get; set; }

        [Required]
        public DateTime SentAt { get; set; } = DateTime.Now;

        [Required]
        public string Content { get; set; }
    }
}